# https://huggingface.co/roneneldan/TinyStories-33M
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig

# help(GenerationConfig)

model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
#help(model)

model.cuda()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
#help(tokenizer)

prompt = "Once upon a time there was"

# help(tokenizer.encode)
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
print(input_ids)

# help(model.generate)
output = model.generate(input_ids, max_length = 1000, num_beams=1,
                        generation_config=GenerationConfig(do_sample=True,temperature=1.))

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)

# What will be the evaluation dataset?
# I guess I should just train for ?an epoch?
# First I should increase the batch size as much as possible
# And pass a lambda x:x as the evaluation function?
# I don't really care about the validation loss, just 
# what the generated text looks like
