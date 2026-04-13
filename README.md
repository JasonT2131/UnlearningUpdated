# Unlearning

LLMs are trained with data that is potentially harmful if it is allowed to respond to the prompt
Example: if LLM is accidentally trained with SSN data, it would be harmful if it gives out the SSN of the public

How do we accurately measure whether a model does not contain certain information anymore. Since there is no 'forgetting' mechanism, the metric measures how 'unwilling' or resistant the model is to regurgitate the adversarial information.

Dataset: TOFU

Models: Mixture of LLAMA models of different strengths

Goal: Use TOFU as the target dataset to forget. Can we make a new metric to measure resistance of answering accurately.
