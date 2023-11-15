from datasets import load_dataset
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
dataset = load_dataset('conv_ai', split="train")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    text = [str(a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(f) for a, b, c, d, e, f in zip(examples['id'], examples['dialogId'], examples['context'], examples['users'], examples['evaluation'], examples['thread'])]
    return tokenizer(text, truncation=True, padding='max_length')





preprocessed_dataset = dataset.map(preprocess_function, batched=True)
training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000, 
        save_total_limit=2,
        max_steps = 2778,
)

model = GPT2LMHeadModel.from_pretrained('gpt2')

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
)

# Initialize the trainer
trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=preprocessed_dataset,
)

# Train the model
trainer.train()

def chat(model, tokenizer):
    while True:
        # Get a message from the user
        message = input("\033[092mUser: \033[0m")

        # Check if the user wants to quit
        if message.lower() == 'quit':
            break

        # Generate a response
        input_ids = tokenizer.encode(message, return_tensors='pt')
        input_ids = input_ids.to(model.device)
        response_ids = model.generate(input_ids)
        response = tokenizer.decode(response_ids[0])

        print("\033[92mModel:\033[0m ", response)

# Call the function to start the chat
chat(model, tokenizer)






