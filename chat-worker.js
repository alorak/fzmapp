// Runs Gemma 4 in the browser for "Makaleyi sor". Answers are generated only from the abstract the page sends.
// Gemma 4 needs Transformers.js 4; the search worker keeps using 3.7.2 and its cached index.
const library = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
let processor;
let model;
let stopping;
let loading;

async function load(modelId, dtype) {
  const {env, AutoProcessor, Gemma4ForCausalLM, InterruptableStoppingCriteria} = await import(library);
  env.allowLocalModels = false;
  let lastUpdate = -Infinity;
  const progress_callback = event => {
    if (event.status !== 'progress_total' || !event.total) return;
    if (performance.now() - lastUpdate < 150) return;
    lastUpdate = performance.now();
    self.postMessage({type: 'progress', loaded: event.loaded, total: event.total});
  };
  processor = await AutoProcessor.from_pretrained(modelId, {progress_callback});
  // Loading through the ...ForCausalLM class fetches only the text parts (embed_tokens and decoder),
  // not the vision and audio encoders.
  model = await Gemma4ForCausalLM.from_pretrained(modelId, {dtype, device: 'webgpu', progress_callback});
  stopping = new InterruptableStoppingCriteria();
  self.postMessage({type: 'status', state: 'warming'});
  // A first short run compiles the WebGPU shaders so the first answer starts sooner.
  const warmup = await processor('a', null, null, {add_special_tokens: false});
  await model.generate({...warmup, max_new_tokens: 1});
}

async function generate(messages) {
  const {TextStreamer} = await import(library);
  // Gemma 4's chat template applies `trim` to the system message content, so content must be a plain string.
  const prompt = processor.apply_chat_template(
    messages.map(({role, content}) => ({role, content: String(content)})),
    {add_generation_prompt: true, enable_thinking: false},
  );
  const inputs = await processor(prompt, null, null, {add_special_tokens: false});
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: text => self.postMessage({type: 'token', text}),
  });
  stopping.reset();
  await model.generate({
    ...inputs,
    do_sample: false,
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
