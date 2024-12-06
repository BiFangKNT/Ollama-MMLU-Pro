# 更适合中国宝宝体质的本地化模板
**适配windows版ollama的本地api，并增加`run_models.bat`模板**

## 开始之前

> 建议使用`anaconda`虚拟环境。进入`anaconda`环境后，cd进项目目录，运行`pip install -r requirements.txt`命令安装环境依赖。

`.bat`模板默认已存在anaconda环境，并且配好依赖。python版本建议`3.10`以上，我的环境是`3.10`。`.bat`模板默认使用`anaconda`的默认环境`base`，如果你打算使用别的环境，自行更改模板中的命令：
```
:: 激活conda环境
call conda activate base
```

# Ollama-MMLU-Pro

This is a modified version of run_gpt4o.py from [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), and it lets you run [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) benchmark via the OpenAI Chat Completion API. It's tested on Ollama 和 Llama.cpp, but it should also work with LMStudio, Koboldcpp, Oobabooga with openai extension, etc.

<a href="https://colab.research.google.com/github/chigkim/Ollama-MMLU-Pro/blob/main/Ollama_MMLU_Pro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

I kept the  testing and scoring  method exactly the same as the original script, adding only a few features to simplify running the test and displaying the results. To see    the exact changes, compare between mmlu-pro branch against main with git diff:

```bash
git diff mmlu-pro..main -- run_openai.py
```

## Usage

Change the config.toml according to your setup.

```shell
pip install -r requirements.txt
python run_openai.py
```

You can also override   settings in configuration file    with  command line flags like --model, ----category, etc. For example, if you   specify `--model phi3`, all the settings  from configuration file will be loaded except model. See `python run_openai.py -h` for more info.

## Additional Notes

* If an answer cannot be extracted from the model's response, the script will randomly assign an answer. It's the same way as the original script.
* The total score represents the number of correct answers out of the total number of attempts including random guess attempts. This is the score from the original script.
* "Random Guess Attempts" indicates the total number of random guesses out of the total attempts.
* "Correct Random Guesses" shows the number of random guesses that were correct out of all the random guesses.
* "Adjusted Score Without Random Guesses" subtracts all random guesses from the correct answers and the total answers.
* The last overall   score in the table  is calculated as: the total number of correct answers across all categories / the total number of all attempts across all categories * 100.
* All the   scores in percentage are rounded numbers.
