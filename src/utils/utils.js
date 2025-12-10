import config from '../config/config.js';
import tokenManager from '../auth/token_manager.js';
import { generateRequestId } from './idGenerator.js';
import os from 'os';

function needsThoughtFlag(modelName = ''){
  return modelName.startsWith('gemini-') && modelName.includes('thinking');
}
function isClaudeThinking(modelName = ''){
  return modelName.includes('claude') && modelName.includes('-thinking');
}

// Claude 思考块：type=thinking；Gemini 思考块：text+thought:true
function buildThinkingPart(modelName, text, enableThinking){
  if (!enableThinking) return null;
  if (isClaudeThinking(modelName)){
    return { type: 'thinking', content: text };
  }
  if (needsThoughtFlag(modelName)){
    return { text, thought: true };
  }
  return null;
}

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
          })
        }
      }
    }
  }
  return result;
}

function sanitizeContent(content, enableThinking, modelName){
  if (typeof content === 'string') return content;
  if (Array.isArray(content)){
    return content.map(item=>{
      // 把上游的 thinking 对象转成文本/思考块
      if (item.thinking){
        const txt = item.thinking.content || item.thinking.text || '';
        const p = buildThinkingPart(modelName, txt, enableThinking);
        return p ?? { text: txt };
      }
      if (item.type === 'thinking'){
        const txt = item.content || item.text || '';
        const p = buildThinkingPart(modelName, txt, enableThinking);
        return p ?? { text: txt };
      }
      if (item.type === 'text'){
        return { text: item.text || '' };
      }
      if (item.type === 'image_url'){
        return item; // 保留 multimodal
      }
      return item;
    });
  }
  if (content && typeof content === 'object' && content.thinking){
    const txt = content.thinking.content || content.thinking.text || '';
    const p = buildThinkingPart(modelName, txt, enableThinking);
    return p ?? { text: txt };
  }
  return content;
}

function handleUserMessage(extracted, antigravityMessages){
  antigravityMessages.push({
    role: "user",
    parts: [
      { text: extracted.text },
      ...extracted.images
    ]
  })
}

function handleAssistantMessage(message, antigravityMessages, modelName, enableThinking){
  const lastMessage = antigravityMessages[antigravityMessages.length - 1];
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasContent = message.content && message.content.trim() !== '';
  const needsThought = needsThoughtFlag(modelName);
  const isClaudeThink = isClaudeThinking(modelName);

  const antigravityTools = hasToolCalls ? message.tool_calls.map(toolCall => ({
    functionCall: {
      id: toolCall.id,
      name: toolCall.function.name,
      args: { query: toolCall.function.arguments }
    },
    ...(needsThought ? { thought: true } : {})
  })) : [];

  const prependThinking = hasToolCalls
    ? buildThinkingPart(modelName, "I will use the tool to process this request.", enableThinking)
    : null;

  if (lastMessage?.role === "model" && hasToolCalls && !hasContent){
    if (prependThinking) lastMessage.parts.push(prependThinking);
    lastMessage.parts.push(...antigravityTools);
  } else {
    const parts = [];

    // Claude thinking: 工具调用时必须先有 thinking block
    if (hasToolCalls && prependThinking){
      parts.push(prependThinking);
    }

    if (hasContent){
      parts.push({ text: message.content.trimEnd() });
    } else if (hasToolCalls && prependThinking){
      // 已经加过思考，不再加文本
    } else if (hasToolCalls && enableThinking && needsThought){
      parts.push({ text: "I will use the tool to process this request.", thought: true });
    }

    parts.push(...antigravityTools);

    antigravityMessages.push({ role: "model", parts });
  }
}

function handleToolCall(message, antigravityMessages){
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
      response: { output: message.content }
    }
  };

  if (lastMessage?.role === "user" && lastMessage.parts.some(p => p.functionResponse)) {
    lastMessage.parts.push(functionResponse);
  } else {
    antigravityMessages.push({ role: "user", parts: [functionResponse] });
  }
}

// 深度清洗 parts，移除/转换 thinking 残留
function deepSanitizeParts(parts, enableThinking, modelName){
  return (parts || []).map(p => {
    if (p.thinking){
      const txt = p.thinking.content || p.thinking.text || '';
      const t = buildThinkingPart(modelName, txt, enableThinking);
      return t ?? { text: txt };
    }
    if (p.type === 'thinking'){
      // Claude thinking 合法：保留
      return isClaudeThinking(modelName) ? p : { text: p.content || p.text || '' };
    }
    if (p.parts){
      return { ...p, parts: deepSanitizeParts(p.parts, enableThinking, modelName) };
    }
    return p;
  });
}

function openaiMessageToAntigravity(openaiMessages, modelName, enableThinking){
  const antigravityMessages = [];
  for (const message of openaiMessages) {
    const sanitized = {
      ...message,
      content: sanitizeContent(message.content, enableThinking, modelName)
    };
    if (sanitized.role === "user" || sanitized.role === "system") {
      const extracted = extractImagesFromContent(sanitized.content);
      handleUserMessage(extracted, antigravityMessages);
    } else if (sanitized.role === "assistant") {
      handleAssistantMessage(sanitized, antigravityMessages, modelName, enableThinking);
    } else if (sanitized.role === "tool") {
      handleToolCall(sanitized, antigravityMessages);
    }
  }
  return antigravityMessages;
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

function isEnableThinking(modelName){
  return modelName.endsWith('-thinking') ||
    modelName === 'gemini-2.5-pro' ||
    modelName.startsWith('gemini-3-pro-') ||
    modelName === "rev19-uic3-1p" ||
    modelName === "gpt-oss-120b-medium"
}

function generateRequestBody(openaiMessages,modelName,parameters,openaiTools,token){
  const enableThinking = isEnableThinking(modelName);
  const actualModelName = modelMapping(modelName);

  const contents = openaiMessageToAntigravity(openaiMessages, modelName, enableThinking);
  const sanitizedContents = contents.map(msg => ({
    ...msg,
    parts: deepSanitizeParts(msg.parts, enableThinking, modelName)
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
        functionCallingConfig: { mode: "VALIDATED" }
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