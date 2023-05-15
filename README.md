# Alpaca dataset with Claude
We used Anthropic's Claude API to generate datasets for instruction tuning. 

## Why Claude
We have made modifications to Stanford's [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) project, which now allows us to use Claude alongside gpt-3.5-turbo. Claude offers a number of advantages over other language models:

1. Claude has the largest context window of any language model available, with a context window size of [100k](https://www.anthropic.com/index/100k-context-windows). This is particularly useful for generating instructions that require a lot of background information, such as strategic wargames.

2. The pricing model for Claude is very generous, with the ability to generate around 50k instructions for free.

Overall, these advantages make Claude a valuable tool for generating high-quality instructions with a large context window and at a reasonable cost.

## How to run
1. export your Anthropic API key with the following command:
```
export ANTHROPIC_API_KEY=sk-ant-api03-........AA
```
2. Run the following command:
```
python -m generate_instruction_pytho generate_instruction_following_data --num_prompt_instructions 3 --num_instructions_to_generate 10000 --client claude
```
## How to customize
While this project was originally designed for generating instructions for military training, it can be easily modified to generate instructions for any domain. Here are the steps to customize the project for your specific needs:

1. Update the [seed tasks](https://github.com/shossain/AlpacaDataCleaned/blob/main/seed_tasks_pytho.jsonl) to reflect your domain. Instead of using military training tasks as a starting point, use tasks that are relevant to your industry or field.

2. Modify the requirements in the [prompts](https://github.com/shossain/AlpacaDataCleaned/blob/main/prompt_pytho_claude.txt) to guide Claude towards your specific problem. By changing the prompts, you can direct the language model to generate instructions that are tailored to your needs.

## Acknowledgments
The original version of the Alpaca dataset was sourced from tatsu-lab's [github repository](https://github.com/tatsu-lab/stanford_alpaca). A cleaned up version with OpenAI's text-davinci-003 was released [here](https://github.com/gururise/AlpacaDataCleaned). 
