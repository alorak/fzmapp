// Runs Qwen3 in the browser for "Makaleyi sor". Answers are generated only from the abstract the page sends.
const library = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';
let tokenizer;
let model;
let stopping;
let loading;

async function load(modelId, dtype) {
  const {env, AutoTokenizer, AutoModelForCausalLM, InterruptableStoppingCriteria} = await import(library);
  env.allowLocalModels = false;
  const files = new Map();
  let lastUpdate = -Infinity;
  const progress_callback = event => {
    if (event.status !== 'progress' || !event.total) return;
    files.set(event.file, {loaded: event.loaded, total: event.total});
    if (performance.now() - lastUpdate < 150) return;
    lastUpdate = performance.now();
    let loaded = 0;
    let total = 0;
    for (const file of files.values()) {
      loaded += file.loaded;
      total += file.total;
    }
    self.postMessage({type: 'progress', loaded, total});
  };
  tokenizer = await AutoTokenizer.from_pretrained(modelId, {progress_callback});
  model = await AutoModelForCausalLM.from_pretrained(modelId, {dtype, device: 'webgpu', progress_callback});
  stopping = new InterruptableStoppingCriteria();
  self.postMessage({type: 'status', state: 'warming'});
  // A first short run compiles the WebGPU shaders so the first answer starts sooner.
  const warmup = tokenizer('a');
  await model.generate({...warmup, max_new_tokens: 1});
}

async function generate(messages) {
  const {TextStreamer} = await import(library);
  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: false,
  });
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: text => self.postMessage({type: 'token', text}),
  });
  stopping.reset();
  await model.generate({
    ...inputs,
    do_sample: true,
    temperature: 0.6,
    top_p: 0.8,
    top_k: 20,
    repetition_penalty: 1.05,
    max_new_tokens: 512,
    streamer,
    stopping_criteria: stopping,
  });
}

self.onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      loading ??= load(data.model, data.dtype);
      await loading;
      self.postMessage({type: 'status', state: 'ready'});
    } else if (data.type === 'generate') {
      await loading;
      await generate(data.messages);
      self.postMessage({type: 'done'});
    } else if (data.type === 'stop') {
      stopping?.interrupt();
    }
  } catch (error) {
    if (data.type === 'load') loading = undefined;
    self.postMessage({type: 'error', message: error.message || String(error), phase: data.type});
  }
};
