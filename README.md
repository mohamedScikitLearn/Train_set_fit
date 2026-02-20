You don't need to make LLM call for most trivial NLP tasks, and 
you don't need thousands of labeled examples to build a text classifier. 

No prompts, no GPU required!! 

I just put together a notebook that trains a sentiment analysis model (just for demo, but could be applied for any given NLP task) with only 10 examples per class — and exports it to ONNX for production-ready inference.
Here's the idea:

The problem: Traditional fine-tuning needs hundreds or thousands of labeled examples, a GPU, and hours of training time. That's expensive and slow when you just need a quick classifier.

The solution: SetFit (Sentence Transformer Fine-tuning by Hugging Face)
30 total training examples (10 positive, 10 negative, 10 neutral)
Trains in ~2 minutes on CPU
No prompts, no GPU required!! 
Competitive accuracy with models trained on 100x more data
It works by generating contrastive pairs from your few examples — pushing same-class embeddings closer together and different-class embeddings apart. Then a simple logistic regression does the classification.
But training is only half the story.

In production, you don't want a 2 GB PyTorch dependency sitting on your server. So the notebook walks through exporting the model to ONNX — a universal model format that runs on ONNX Runtime (~50 MB) with automatic graph optimizations.

The export breaks the model into two parts:
The transformer body (heavy compute) → exported to .onnx
The classification head (lightweight sklearn model) → pickled separately
Result: a production inference pipeline that needs zero PyTorch, runs 2-5x faster, and deploys anywhere — Linux, Windows, ARM, even WebAssembly.
Three files. Two pip packages. 

That's your entire deployment.
The notebook covers everything end to end: data prep, training, evaluation, ONNX export with torch.onnx.export, inference with onnxruntime, and a PyTorch vs ONNX speed benchmark.
