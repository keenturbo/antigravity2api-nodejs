import config from '../config/config.js';
import tokenManager from '../auth/token_manager.js';
import { generateRequestId } from './idGenerator.js';
import os from 'os';

function extractImagesFromContent(content) {
  const result = { text: '', images: [] };

  // 如果content是字符串，直接返回
  if (typeof content === 'string') {
    result.text = content;
    return result;
  }

  // 如果content是数组（multimodal格式）
  if (Array.isArray(content)) {
    for (const item of content) {
      if (item.type === 'text') {
        result.text += item.text;
      } else if (item.type === 'image_url') {
        // 提取base64图片数据
        const imageUrl = item.image_url?.url || '';

        // 匹配 data:image/{format};base64,{data} 格式
        const match = imageUrl.match(/^data:image\/(\w+);base64,(.+)$/);
        if (match) {
          const format = match[1]; // 例如 png, jpeg, jpg
          const base64Data = match[2];
          result.images.push({
            inlineData: {
              mimeType: `image/${format}`,
              data: base64Data
            }
          })
        }
      }
    }
  }

  return result;
}

// 深度消毒：移除/转换任何 thinking 字段，统一为 text/标准 multimodal 结构
function sanitizeContent(rawContent){
  if (!rawContent) return rawContent;

  // 直接是 thinking 对象
  if (rawContent && typeof rawContent === 'object' && rawContent.thinking) {
    return rawContent.thinking.content || '';
  }

  // 数组形式的内容
  if (Array.isArray(rawContent)) {
    const parts = [];
    for (const item of rawContent) {
      if (!item) continue;
      if (item.thinking) {
        parts.push({ type: 'text', text: item.thinking.content || '' });
      } else if (item.type === 'thinking') {
        parts.push({ type: 'text', text: item.content || item.text || '' });
      } else if (item.type === 'text' && typeof item.text === 'string') {
        parts.push({ type: 'text', text: item.text });
      } else if (item.type === 'image_url') {
        parts.push(item);
      } else if (typeof item.text === 'string') {
        parts.push({ type: 'text', text: item.text });
      }
    }
    return parts.length > 0 ? parts : '';
  }

  return rawContent;
}

function handleUserMessage(extracted, antigravityMessages){
  antigravityMessages.push({
    role: "user",
    parts: [
      {
        text: extracted.text
      },
      ...extracted.images
    ]
  })
}

// 统一思考格式：所有思考模型都用 Gemini 兼容的 { text, thought: true }
function buildThinkingPart(text){
  return { text, thought: true };
}

// 仅在 needsThought=true 时注入思考；
// Gemini Thinking 需要 thought 标记，Claude 等非 Gemini 即便是 thinking 模型也不用 thought。
function handleAssistantMessage(message, antigravityMessages, modelName, needsThought){
  const lastMessage = antigravityMessages[antigravityMessages.length - 1];
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const rawContent = sanitizeContent(message.content);

  // 兼容数组文本内容
  const textContent = Array.isArray(rawContent)
    ? rawContent.filter(p => p?.type === 'text' && typeof p.text === 'string').map(p => p.text).join('')
    : rawContent;
  const hasContent = typeof textContent === 'string' && textContent.trim() !== '';
  const defaultThoughtText = "I will use the tool to process this request.";

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

  const maybeThought = needsThought ? buildThinkingPart(defaultThoughtText) : null;

  if (lastMessage?.role === "model" && hasToolCalls && !hasContent){
    if (maybeThought) lastMessage.parts.push(maybeThought);
    lastMessage.parts.push(...antigravityTools);
  }else{
    const parts = [];
    if (needsThought) {
      if (hasToolCalls && !hasContent) {
        if (maybeThought) parts.push(maybeThought);
      } else if (hasContent) {
        parts.push({ text: textContent.trimEnd() });
      }
    } else {
      if (hasContent) {
        parts.push({ text: textContent.trimEnd() });
      }
    }
    parts.push(...antigravityTools);

    antigravityMessages.push({
      role: "model",
      parts
    })
  }
}

function handleToolCall(message, antigravityMessages){
  // 从之前的 model 消息中找到对应的 functionCall name
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
  
  // 如果上一条消息是 user 且包含 functionResponse，则合并
  if (lastMessage?.role === "user" && lastMessage.parts.some(p => p.functionResponse)) {
    lastMessage.parts.push(functionResponse);
  } else {
    antigravityMessages.push({
      role: "user",
      parts: [functionResponse]
    });
  }
}

// 判定是否需要 thought 标记（只针对 Gemini 思考模型）
function needsThoughtFlag(modelName){
  return modelName === 'gemini-2.5-pro' ||
    modelName.startsWith('gemini-3-pro-') ||
    modelName === 'gemini-2.5-flash-thinking';
}

function openaiMessageToAntigravity(openaiMessages, modelName, needsThought){
  const antigravityMessages = [];
  for (const message of openaiMessages) {
    // 消毒：把外部传入的 thinking 对象转换为纯 text / 标准 multimodal
    const sanitizedContent = sanitizeContent(message.content);
    const msg = { ...message, content: sanitizedContent };

    if (msg.role === "user" || msg.role === "system") {
      const extracted = extractImagesFromContent(msg.content);
      handleUserMessage(extracted, antigravityMessages);
    } else if (msg.role === "assistant") {
      handleAssistantMessage(msg, antigravityMessages, modelName, needsThought);
    } else if (msg.role === "tool") {
      handleToolCall(msg, antigravityMessages);
    }
  }
  
  return antigravityMessages;
}

// 终极消毒：确保 parts 树中不残留 thinking 字段
function deepSanitizeParts(parts, needsThought) {
  if (!parts) return [];
  const cleaned = [];
  for (const part of parts) {
    if (!part) continue;
    if (part.thinking) {
      cleaned.push({ text: part.thinking.content || '', ...(needsThought ? { thought: true } : {}) });
      continue;
    }
    if (part.type === 'thinking') {
      cleaned.push({ text: part.content || part.text || '', ...(needsThought ? { thought: true } : {}) });
      continue;
    }
    const clone = { ...part };
    if (clone.thinking) delete clone.thinking;
    if (clone.parts) clone.parts = deepSanitizeParts(clone.parts, needsThought);
    cleaned.push(clone);
  }
  return cleaned;
}

function generateGenerationConfig(parameters, enableThinking, actualModelName){
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
  }
  if (enableThinking && actualModelName.includes("claude")){
    delete generationConfig.topP;
  }
  return generationConfig
}
function convertOpenAIToolsToAntigravity(openaiTools){
  if (!openaiTools || openaiTools.length === 0) return [];
  return openaiTools.map((tool)=>{
    delete tool.function.parameters.$schema;
    return {
      functionDeclarations: [
        {
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters
        }
      ]
    }
  })
}

function modelMapping(modelName){
  if (modelName === "claude-sonnet-4-5-thinking"){
    return "claude-sonnet-4-5";
  } else if (modelName === "claude-opus-4-5"){
    return "claude-opus-4-5-thinking";
  } else if (modelName === "gemini-2.5-flash-thinking"){
    return "gemini-2.5-flash";
  }
  return modelName;
}

// 是否允许思考（决定 generationConfig.thinkingConfig），但思考标记仅在 needsThoughtFlag 为真时加。
function isEnableThinking(modelName){
  return modelName.endsWith('-thinking') ||
    modelName === 'gemini-2.5-pro' ||
    modelName.startsWith('gemini-3-pro-') ||
    modelName === "rev19-uic3-1p" ||
    modelName === "gpt-oss-120b-medium"
}

function generateRequestBody(openaiMessages,modelName,parameters,openaiTools,token){
  // 只对 Gemini 思考模型加 thought 标记，其余即便带 thinking 后缀也不加 thought。
  const needsThought = needsThoughtFlag(modelName);
  // 思考开关用于 generationConfig；若你希望非 Gemini thinking 也带 includeThoughts，可保留 isEnableThinking；否则可改为 needsThought。
  const enableThinking = isEnableThinking(modelName);
  const actualModelName = modelMapping(modelName);
  const contents = openaiMessageToAntigravity(openaiMessages, modelName, needsThought);
  const sanitizedContents = contents.map(msg => ({
    ...msg,
    parts: deepSanitizeParts(msg.parts, needsThought)
  }));
  
  return{
    project: token.projectId,
    requestId: generateRequestId(),
    request: {
      contents: sanitizedContents,
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
  }
}
function getDefaultIp(){
  const interfaces = os.networkInterfaces();
  if (interfaces.WLAN){
    for (const inter of interfaces.WLAN){
      if (inter.family === 'IPv4' && !inter.internal){
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
export{
  generateRequestId,
  generateRequestBody,
  getDefaultIp
}
