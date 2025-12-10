import config from '../config/config.js';
import tokenManager from '../auth/token_manager.js';
import { generateRequestId } from './idGenerator.js';
import os from 'os';

function extractImagesFromContent(content) {
  const result = { text: '', images: [] };
  if (typeof content === 'string') {
    result.text = content;
    return result;
  }
  if (Array.isArray(content)) {
    for (const item of content) {
      if (item.type === 'text') {
        result.text += item.text;
      } else if (item.type === 'image_url') {
        const imageUrl = item.image_url?.url || '';
        const match = imageUrl.match(/^data:image\/(\w+);base64,(.+)$/);
        if (match) {
          const format = match[1];
          const base64Data = match[2];
          result.images.push({
            inlineData: {
              mimeType: `image/${format}`,
              data: base64Data
            }
          });
        }
      }
    }
  }
  return result;
}

function handleUserMessage(extracted, antigravityMessages) {
  antigravityMessages.push({
    role: "user",
    parts: [
      { text: extracted.text },
      ...extracted.images
    ]
  });
}

// 仅 Gemini 需要 thought:true
function needsThoughtFlag(modelName = '') {
  return modelName.startsWith('gemini-');
}

// Claude 思考模型判定（仅用于去除 thought，不做特殊字段）
function isClaudeThinking(modelName = '') {
  return modelName.startsWith('claude') && modelName.endsWith('-thinking');
}

const defaultThoughtText = "Thought: I need to use the tool to fulfill the request.";

function sanitizeIncomingContent(content) {
  // 去掉上游可能带入的 thinking 字段
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map(item => {
      if (item.thinking) {
        return { text: item.thinking.content || item.thinking.text || '' };
      }
      if (item.type === 'thinking') {
        return { text: item.content || item.text || '' };
      }
      if (item.type === 'text') return { text: item.text || '' };
      return item; // image_url 或其它保持
    });
  }
  if (content && typeof content === 'object' && content.thinking) {
    return { text: content.thinking.content || content.thinking.text || '' };
  }
  return content;
}

// 核心：按模型决定是否加 thought，以及在无 content 有工具时补思考文本
function handleAssistantMessage(message, antigravityMessages, modelName) {
  const lastMessage = antigravityMessages[antigravityMessages.length - 1];
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasContent = message.content && message.content.trim() !== '';
  const needsThought = needsThoughtFlag(modelName);
  const isClaudeThink = isClaudeThinking(modelName);

  const antigravityTools = hasToolCalls ? message.tool_calls.map(toolCall => ({
    functionCall: {
      id: toolCall.id,
      name: toolCall.function.name,
      args: {
        query: toolCall.function.arguments
      }
    },
    ...(needsThought ? { thought: true } : {})
  })) : [];

  if (lastMessage?.role === "model" && hasToolCalls && !hasContent) {
    // 补思考文本，再追加工具
    lastMessage.parts.push(
      needsThought
        ? { text: defaultThoughtText, thought: true }
        : { text: defaultThoughtText },
      ...antigravityTools
    );
  } else {
    const parts = [];
    if (hasToolCalls && !hasContent) {
      parts.push(
        needsThought
          ? { text: defaultThoughtText, thought: true }
          : { text: defaultThoughtText }
      );
    } else if (hasContent) {
      parts.push({ text: message.content.trimEnd() });
    }
    parts.push(...antigravityTools);

    antigravityMessages.push({
      role: "model",
      parts
    });
  }
}

function handleToolCall(message, antigravityMessages) {
  let functionName = '';
  for (let i = antigravityMessages.length - 1; i >= 0; i--) {
    if (antigravityMessages[i].role === 'model') {
      const parts = antigravityMessages[i].parts;
      for (const part of parts) {
        if (part.functionCall && part.functionCall.id === message.tool_call_id) {
          functionName = part.functionCall.name;
          break;
        }
      }
      if (functionName) break;
    }
  }

  const lastMessage = antigravityMessages[antigravityMessages.length - 1];
  const functionResponse = {
    functionResponse: {
      id: message.tool_call_id,
      name: functionName,
      response: {
        output: message.content
      }
    }
  };

  if (lastMessage?.role === "user" && lastMessage.parts.some(p => p.functionResponse)) {
    lastMessage.parts.push(functionResponse);
  } else {
    antigravityMessages.push({
      role: "user",
      parts: [functionResponse]
    });
  }
}

function openaiMessageToAntigravity(openaiMessages, modelName) {
  const antigravityMessages = [];
  for (const message of openaiMessages) {
    // 先消毒上游 content 中的 thinking 字段
    const sanitizedContent = sanitizeIncomingContent(message.content);

    if (message.role === "user" || message.role === "system") {
      const extracted = extractImagesFromContent(sanitizedContent);
      handleUserMessage(extracted, antigravityMessages);
    } else if (message.role === "assistant") {
      const sanitizedMessage = { ...message, content: sanitizedContent };
      handleAssistantMessage(sanitizedMessage, antigravityMessages, modelName);
    } else if (message.role === "tool") {
      const sanitizedMessage = { ...message, content: sanitizedContent };
      handleToolCall(sanitizedMessage, antigravityMessages);
    }
  }
  return antigravityMessages;
}

function generateGenerationConfig(parameters, enableThinking, actualModelName) {
  const generationConfig = {
    topP: parameters.top_p ?? config.defaults.top_p,
    topK: parameters.top_k ?? config.defaults.top_k,
    temperature: parameters.temperature ?? config.defaults.temperature,
    candidateCount: 1,
    maxOutputTokens: parameters.max_tokens ?? config.defaults.max_tokens,
    stopSequences: [
      "<|user|>",
      "<|bot|>",
      "<|context_request|>",
      "<|endoftext|>",
      "<|end_of_turn|>"
    ],
    thinkingConfig: {
      includeThoughts: enableThinking,
      thinkingBudget: enableThinking ? 1024 : 0
    }
  };
  if (enableThinking && actualModelName.includes("claude")) {
    delete generationConfig.topP;
  }
  return generationConfig;
}

function convertOpenAIToolsToAntigravity(openaiTools) {
  if (!openaiTools || openaiTools.length === 0) return [];
  return openaiTools.map((tool) => {
    delete tool.function.parameters.$schema;
    return {
      functionDeclarations: [
        {
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters
        }
      ]
    };
  });
}

function modelMapping(modelName) {
  if (modelName === "claude-sonnet-4-5-thinking") {
    return "claude-sonnet-4-5";
  } else if (modelName === "claude-opus-4-5") {
    return "claude-opus-4-5-thinking";
  } else if (modelName === "gemini-2.5-flash-thinking") {
    return "gemini-2.5-flash";
  }
  return modelName;
}

// 启用思考的模型集合（Claude thinking 只在 generationConfig 层影响，不加 thought 标记）
function isEnableThinking(modelName) {
  return modelName.endsWith('-thinking') ||
    modelName === 'gemini-2.5-pro' ||
    modelName.startsWith('gemini-3-pro-') ||
    modelName === "rev19-uic3-1p" ||
    modelName === "gpt-oss-120b-medium";
}

function generateRequestBody(openaiMessages, modelName, parameters, openaiTools, token) {
  const enableThinking = isEnableThinking(modelName);
  const actualModelName = modelMapping(modelName);

  return {
    project: token.projectId,
    requestId: generateRequestId(),
    request: {
      contents: openaiMessageToAntigravity(openaiMessages, modelName),
      systemInstruction: {
        role: "user",
        parts: [{ text: config.systemInstruction }]
      },
      tools: convertOpenAIToolsToAntigravity(openaiTools),
      toolConfig: {
        functionCallingConfig: {
          mode: "VALIDATED"
        }
      },
      generationConfig: generateGenerationConfig(parameters, enableThinking, actualModelName),
      sessionId: token.sessionId
    },
    model: actualModelName,
    userAgent: "antigravity"
  };
}

function getDefaultIp() {
  const interfaces = os.networkInterfaces();
  if (interfaces.WLAN) {
    for (const inter of interfaces.WLAN) {
      if (inter.family === 'IPv4' && !inter.internal) {
        return inter.address;
      }
    }
  } else if (interfaces.wlan2) {
    for (const inter of interfaces.wlan2) {
      if (inter.family === 'IPv4' && !inter.internal) {
        return inter.address;
      }
    }
  }
  return '127.0.0.1';
}

export {
  generateRequestId,
  generateRequestBody,
  getDefaultIp
}