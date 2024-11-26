import os
import re
import json
import time
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import codecs
import toml
import argparse
import queue
import numpy as np
import copy
import requests
import subprocess
import atexit
import signal

parser = argparse.ArgumentParser(
	prog="python3 run_openai.py",
	description="Run MMLU Pro Benchmark for  a local LLM  via  OpenAI Compatible API.",
	epilog="Specify  options above  to override  one or more settings from config.",
)
parser.add_argument(
	"-c",
	"--config",
	help="Configuration file. Default=config.toml",
	default="config.toml",
)
parser.add_argument(
	"-u",
	"--url",
	help="server url",
)
parser.add_argument("-a", "--api", help="api key")
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument(
	"--timeout",
	type=float,
	help="Request timeout in seconds",
)
parser.add_argument("--category", type=str)
parser.add_argument("-p", "--parallel", type=int, help="Number of parallel requests")
parser.add_argument("-v", "--verbosity", type=int, help="Verbosity level 0-2")
parser.add_argument(
	"--log_prompt",
	help="Writes exact prompt and response into log.txt",
	action="store_true",
)
parser.add_argument(
	"--comment", type=str, help="Comment to be included in the final report."
)
args = parser.parse_args()
config = toml.load(open(args.config))
if args.url:
	config["server"]["url"] = args.url
if args.api:
	config["server"]["api_key"] = args.api
if args.model:
	config["server"]["model"] = args.model
if args.timeout:
	config["server"]["timeout"] = args.timeout
if args.category:
	config["test"]["categories"] = [args.category]
if args.parallel:
	config["test"]["parallel"] = args.parallel
if args.verbosity:
	config["log"]["verbosity"] = args.verbosity
if args.log_prompt:
	config["log"]["log_prompt"] = args.log_prompt
if args.comment:
	config["comment"] = args.comment


client = OpenAI(
	base_url=config["server"]["url"],
	api_key=config["server"]["api_key"],
	timeout=config["server"]["timeout"],
)


def log(message):
	print(message)
	with codecs.open(log_path, "a", "utf-8") as file:
		file.write(message + "\n")


def get_chat_completion(messages):
	try:
		prompt = ""
		for msg in messages:
			role = msg["role"]
			content = msg["content"]
			if role == "system":
				prompt += f"System: {content}\n"
			elif role == "user":
				prompt += f"User: {content}\n"
			elif role == "assistant":
				prompt += f"Assistant: {content}\n"
		
		url = f"{config['server']['url'].rstrip('/')}"
		data = {
			"model": config["server"]["model"],
			"prompt": prompt,
			"stream": False,
			"options": {
				"temperature": config["inference"]["temperature"],
				"num_predict": config["inference"]["max_tokens"],
			}
		}
		
		response = requests.post(url, json=data)
		response_json = response.json()
		
		if "response" not in response_json:
			print(f"API返回异常: {response_json}")
			raise ValueError(f"API返回中缺少response字段: {response_json}")
			
		return response_json["response"].strip()
		
	except Exception as e:
		print(f"重新提交, 错误: {e}")
		time.sleep(3)
		return get_chat_completion(messages)


def get_completion(prompt):
	try:
		response = client.completions.create(
			model=config["server"]["model"],
			prompt=prompt,
			temperature=config["inference"]["temperature"],
			max_tokens=config["inference"]["max_tokens"],
			top_p=config["inference"]["top_p"],
			frequency_penalty=0,
			presence_penalty=0,
			stop=["Question:"],
			timeout=config["server"]["timeout"],
		)
		try:
			usage_q.put(
				(response.usage.prompt_tokens, response.usage.completion_tokens)
			)
		except:
			pass
		if response.choices:
			return response.choices[0].text.strip()
		elif response.content:
			return response.content.strip()
		print("Can't get response.")
		return None
	except Exception as e:
		print("Resubmitting, Error: ", e)
		time.sleep(3)
		return get_completion(prompt)


def load_mmlu_pro():
	dataset = load_dataset("TIGER-Lab/MMLU-Pro")
	test_df, val_df = dataset["test"], dataset["validation"]
	test_df = preprocess(test_df)
	val_df = preprocess(val_df)
	return test_df, val_df


def preprocess(test_df):
	res_df = []
	for each in test_df:
		options = []
		for opt in each["options"]:
			if opt == "N/A":
				continue
			options.append(opt)
		each["options"] = options
		res_df.append(each)
	res = {}
	for each in res_df:
		if each["category"] not in res:
			res[each["category"]] = []
		res[each["category"]].append(each)
	return res


def format_example(question, options, cot_content=""):
	if cot_content == "":
		cot_content = "Let's think step by step."
	if cot_content.startswith("A: "):
		cot_content = cot_content[3:]
	example = "Question: {}\nOptions: ".format(question)
	choice_map = "ABCDEFGHIJ"
	for i, opt in enumerate(options):
		example += "{}. {}\n".format(choice_map[i], opt)
	return example.strip(), cot_content.strip()


def multi_chat_prompt(cot_examples, question, options):
	messages = [
		{
			"role": "system",
			"content": config["inference"]["system_prompt"],
		},
	]
	for each in cot_examples:
		example, cot_content = format_example(
			each["question"], each["options"], each["cot_content"]
		)
		messages.append({"role": "user", "content": example})
		messages.append({"role": "assistant", "content": "Answer: " + cot_content})
	example, cot_content = format_example(question, options)
	messages.append({"role": "user", "content": example})
	return messages


def single_chat_prompt(cot_examples, question, options):
	messages = [
		{
			"role": "system",
			"content": config["inference"]["system_prompt"],
		},
	]
	prompt = no_chat_prompt(cot_examples, question, options, no_system=True)
	messages.append({"role": "user", "content": prompt})
	return messages


def no_chat_prompt(cot_examples, question, options, no_system=False):
	prompt = config["inference"]["system_prompt"] + "\n\n"
	if no_system:
		prompt = ""
	for each in cot_examples:
		example, cot_content = format_example(
			each["question"], each["options"], each["cot_content"]
		)
		prompt += example + "\n"
		prompt += "Answer: " + cot_content + "\n\n"
	example, cot_content = format_example(question, options)
	prompt += example + "\n"
	prompt += "Answer: " + cot_content
	return prompt


def extract_answer(text):
	pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
	match = re.search(pattern, text)
	if match:
		return match.group(1)
	else:
		return extract_again(text)


def extract_again(text):
	pattern = r".*[aA]nswer:\s*\(?([A-J])\)?"
	match = re.search(pattern, text)
	if match:
		return match.group(1)
	else:
		return extract_final(text)


def extract_final(text):
	pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
	match = re.search(pattern, text, re.DOTALL)
	if match:
		return match[0]
	else:
		if config["log"]["verbosity"] >= 1:
			print("Extraction failed:\n", text)
		return None


def run_single_question(single_question, cot_examples_dict, exist_result):
	exist = True
	q_id = single_question["question_id"]
	for each in exist_result:
		if (
			q_id == each["question_id"]
			and single_question["question"] == each["question"]
		):
			if config["log"]["verbosity"] >= 1:
				print("already exists, skipping.")
			return None, None, None, exist
	exist = False
	category = single_question["category"]
	cot_examples = cot_examples_dict[category]
	question = single_question["question"]
	options = single_question["options"]
	try:
		if config["inference"]["style"] == "single_chat":
			prompt = single_chat_prompt(cot_examples, question, options)
			response = get_chat_completion(prompt)
		elif config["inference"]["style"] == "multi_chat":
			prompt = multi_chat_prompt(cot_examples, question, options)
			response = get_chat_completion(prompt)
		elif config["inference"]["style"] == "no_chat":
			prompt = no_chat_prompt(cot_examples, question, options)
			response = get_completion(prompt)
	except Exception as e:
		print("error", e)
		return None, None, None, exist
	pred = extract_answer(response)
	return prompt, response, pred, exist


def update_result(output_res_path, lock):
	category_record = {}
	res = []
	success = False
	while not success:
		try:
			if os.path.exists(output_res_path):
				with lock:
					with open(output_res_path, "r") as fi:
						res = json.load(fi)
						for each in res:
							category = each["category"]
							if category not in category_record:
								category_record[category] = {"corr": 0.0, "wrong": 0.0}
								category_record["random"] = {"corr": 0.0, "wrong": 0.0}
							if not each["pred"]:
								random.seed(12345)
								x = random.randint(0, len(each["options"]) - 1)
								if x == each["answer_index"]:
									category_record[category]["corr"] += 1
									category_record["random"]["corr"] += 1
								else:
									category_record[category]["wrong"] += 1
									category_record["random"]["wrong"] += 1
							elif each["pred"] == each["answer"]:
								category_record[category]["corr"] += 1
							else:
								category_record[category]["wrong"] += 1
			success = True
		except Exception as e:
			print("Error", e)
	return res, category_record


def evaluate(subjects):
	test_df, dev_df = load_mmlu_pro()
	if not subjects:
		subjects = list(test_df.keys())
	print("assigned subjects", subjects)
	lock = threading.Lock()
	system_prompt = config["inference"]["system_prompt"]
	
	for subject in subjects:
		start = time.time()
		print(f"Testing {subject}...")
		config["inference"]["system_prompt"] = system_prompt.replace(
			"{subject}", subject
		)
		test_data = test_df[subject]
		output_res_path = os.path.join(output_dir, subject + "_result.json")
		output_summary_path = os.path.join(output_dir, subject + "_summary.json")
		
		# 加载已有结果
		res, category_record = update_result(output_res_path, lock)
		
		# 筛选出未完成的题目
		remaining_questions = []
		for q in test_data:
			if not any(r["question_id"] == q["question_id"] for r in res):
				remaining_questions.append(q)

		with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
			futures = {
				executor.submit(run_single_question, each, dev_df, res): each
				for each in remaining_questions
			}
			
			for future in tqdm(
				as_completed(futures), 
				total=len(remaining_questions),
				smoothing=0.0,
				ascii=True,
				desc=f"Progress",
				bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
			):
				each = futures[future]
				label = each["answer"]
				category = subject
				prompt, response, pred, exist = future.result()
				
				if exist:
					continue
					
				if response is not None:
					res, category_record = update_result(output_res_path, lock)
					if category not in category_record:
						category_record[category] = {"corr": 0.0, "wrong": 0.0}
					if config["log"]["log_prompt"]:
						each["prompt"] = prompt
					each["response"] = response
					each["pred"] = pred
					res.append(each)
					
					if config["log"]["verbosity"] >= 2:
						log_json = {
							"id": each["question_id"],
							"question": each["question"],
							"response": each["response"],
							"pred": each["pred"],
							"answer": each["answer"],
						}
						print("\n" + json.dumps(log_json, indent="\t"))
						
					# 更新速度和剩余时间计算
					current_time = time.time() - start
					current_count = len([f for f in futures if f.done()])
					total_remaining = len(remaining_questions) - current_count
					speed = current_count / current_time if current_time > 0 else 0
					remaining = total_remaining / speed if speed > 0 else 0
					
					tqdm.write(f"\rSpeed: {speed:.2f} q/s, Estimated remaining time: {remaining:.2f}s")
					
					if pred is not None:
						if pred == label:
							category_record[category]["corr"] += 1
						else:
							category_record[category]["wrong"] += 1
					else:
						category_record[category]["wrong"] += 1
						
					save_res(res, output_res_path, lock)
					save_summary(category_record, output_summary_path, lock)
					res, category_record = update_result(output_res_path, lock)

		save_res(res, output_res_path, lock)
		log(f"Finished testing {subject} in {elapsed(start)}.")
		save_summary(category_record, output_summary_path, lock, report=True)


def save_res(res, output_res_path, lock):
	temp = []
	exist_q_id = []
	for each in res:
		if each["question_id"] not in exist_q_id:
			exist_q_id.append(each["question_id"])
			temp.append(each)
		else:
			continue
	res = temp
	with lock:
		with open(output_res_path, "w") as fo:
			fo.write(json.dumps(res, indent="\t"))


def print_score(label, corr, wrong, file=None, return_str=False):
	try:
		corr = int(corr)
		wrong = int(wrong)
		total = corr + wrong
		acc = corr / total * 100
		
		message = f"{label}, {corr}/{total}, {acc:.2f}%"
		
		if return_str:
			return message
		if file:
			file.write(message + "\n")
		log(message)
	except Exception as e:
		message = f"{label}, {e} error"
		if return_str:
			return message
		if file:
			file.write(message + "\n")
		log(message)


SUMMARY_SEPARATOR = "\n=== Summary ===\n"

def save_summary(category_record, output_summary_path, lock, report=False):
	total_corr = 0.0
	total_wrong = 0.0
	for k, v in category_record.items():
		if k == "total" or k == "random":
			continue
		cat_acc = v["corr"] / (v["corr"] + v["wrong"])
		category_record[k]["acc"] = cat_acc
		total_corr += v["corr"]
		total_wrong += v["wrong"]
	acc = total_corr / (total_corr + total_wrong)
	category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
	
	if report:
		# 生成当前科目报告内容
		report_content = []
		report_content.append(f"{datetime.now()}")
		
		# 添加配置信息
		config_copy = copy.deepcopy(config)
		del config_copy["server"]["api_key"]
		if "categories" in config_copy["test"]:
			del config_copy["test"]["categories"]
		report_content.append(json.dumps(config_copy, indent="\t"))
		
		# 添加统计信息
		stats = []
		stats.append(print_score("Total", total_corr, total_wrong, return_str=True))
		if "random" in category_record:
			random_corr = category_record["random"]["corr"]
			random_wrong = category_record["random"]["wrong"]
			stats.append(print_score(
				"Random Guess Attempts",
				random_corr + random_wrong,
				total_corr + total_wrong - random_corr - random_wrong,
				return_str=True
			))
			stats.append(print_score("Correct Random Guesses", random_corr, random_wrong, return_str=True))
			stats.append(print_score(
				"Adjusted Score Without Random Guesses",
				total_corr - random_corr,
				total_wrong - random_wrong,
				return_str=True
			))
		report_content.extend(stats)
		
		report_text = "\n".join(report_content) + "\n"
		
		# 写入单独科目报告
		subject_report_path = output_summary_path.replace("_summary.json", "_report.txt")
		with lock:
			with open(subject_report_path, "w", encoding="utf-8") as fo:
				fo.write(report_text)
		
		# 更新总报告
		update_main_report(report_text, lock)

def update_main_report(new_content, lock):
	"""更新总报告，保持总结部分在最后"""
	with lock:
		# 读取现有报告内容
		existing_content = ""
		summary_content = ""
		if os.path.exists(log_path):
			with open(log_path, "r", encoding="utf-8") as f:
				full_content = f.read()
				if SUMMARY_SEPARATOR in full_content:
					# 如果存在总结部分，分离内容
					existing_content, summary_content = full_content.split(SUMMARY_SEPARATOR, 1)
				else:
					existing_content = full_content
		
		# 添加新内容
		updated_content = existing_content + new_content + "\n"
		
		# 生成新的总结
		new_summary = generate_summary(updated_content)
		
		# 写入完整内容
		with open(log_path, "w", encoding="utf-8") as f:
			f.write(updated_content)
			f.write(SUMMARY_SEPARATOR)
			f.write(new_summary)

def generate_summary(content):
	"""从报告内容中提取所有科目成绩并生成总结"""
	# 用于存储每个科目的成绩
	subject_scores = {}
	
	# 解析内容提取成绩
	for line in content.split('\n'):
		if line.startswith("Total, "):
			# 从上下文确定当前科目
			context_lines = content.split('\n')
			for i, ctx_line in enumerate(context_lines):
				if "testing" in ctx_line and "in" in ctx_line:
					subject = ctx_line.split("testing")[1].split("in")[0].strip()
					score = float(line.split(",")[2].strip().rstrip("%"))
					subject_scores[subject] = score
					break
	
	# 生成总结文本
	summary = ["Summary Report", "-" * 20]
	summary.append(f"Total Subjects: {len(subject_scores)}")
	summary.append("\nSubject Scores:")
	for subject, score in subject_scores.items():
		summary.append(f"{subject}: {score:.2f}%")
	
	if subject_scores:
		avg_score = sum(subject_scores.values()) / len(subject_scores)
		summary.append(f"\nAverage Score: {avg_score:.2f}%")
	
	# 生成Markdown表格
	headers = ["overall"] + list(subject_scores.keys())
	separators = ["-" * len(h) for h in headers]
	scores = [f"{sum(subject_scores.values()) / len(subject_scores):.2f}"] + [f"{score:.2f}" for score in subject_scores.values()]
	
	summary.append("\nMarkdown Table:")
	summary.append("| " + " | ".join(headers) + " |")
	summary.append("| " + " | ".join(separators) + " |")
	summary.append("| " + " | ".join(scores) + " |")
	
	return "\n".join(summary) + "\n"


def final_report(assigned_subjects):
	total_corr = 0.0
	total_wrong = 0.0
	random_corr = 0.0
	random_wrong = 0.0
	names = ["overall"]
	scores = []
	valid_subjects = []
	
	# 检查每个科目的结果文件
	for subject in assigned_subjects:
		report_path = os.path.join(output_dir, f"{subject}_report.txt")  # 改用 _report.txt
		if os.path.exists(report_path):
			try:
				# 解析 report.txt 文件内容
				with open(report_path, 'r') as f:
					content = f.read()
					# 提取总分信息
					total_match = re.search(r'Total, (\d+)/(\d+)', content)
					if total_match:
						cat_corr = int(total_match.group(1))
						total = int(total_match.group(2))
						cat_wrong = total - cat_corr
						
						total_corr += cat_corr
						total_wrong += cat_wrong
						scores.append(cat_corr / total)
						names.append(subject)
						valid_subjects.append(subject)
						
					# 提取随机猜测信息
					random_match = re.search(r'Random Guess Attempts, (\d+)/(\d+)', content)
					if random_match:
						random_total = int(random_match.group(1))
						correct_match = re.search(r'Correct Random Guesses, (\d+)/(\d+)', content)
						if correct_match:
							random_corr += int(correct_match.group(1))
							random_wrong += random_total - int(correct_match.group(1))
							
			except Exception as e:
				print(f"读取{subject}结果文件失败: {e}")
				continue
	
	# ... 后续生成报告的代码保持不变 ...


def elapsed(start):
	duration = time.time() - start
	duration_td = timedelta(seconds=duration)
	days = duration_td.days
	hours, remainder = divmod(duration_td.seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	dur_str = ""
	if days:
		dur_str = f"{days} days "
	if hours:
		dur_str += f"{hours} hours "
	if minutes:
		dur_str += f"{minutes} minutes "
	if seconds:
		dur_str += f"{seconds} seconds"
	return dur_str


def token_report():
	ptoks = []
	ctoks = []
	while not usage_q.empty():
		usage = usage_q.get()
		ptoks.append(usage[0])
		ctoks.append(usage[1])
	if ptoks and ctoks:
		log("Token Usage:")
		duration = end - start
		ptoks = np.array(ptoks)
		ctoks = np.array(ctoks)
		log(
			f"Prompt tokens: min {ptoks.min()}, average {ptoks.mean():.0f}, max {ptoks.max()}, total {ptoks.sum()}, tk/s {ptoks.sum()/duration:.2f}"
		)
		log(
			f"Completion tokens: min {ctoks.min()}, average {ctoks.mean():.0f}, max {ctoks.max()}, total {ctoks.sum()}, tk/s {ctoks.sum()/duration:.2f}"
		)


def check_running_models(target_model):
	"""检查运行中的模型并决定是否需要启动目标模型"""
	try:
		result = subprocess.run(['ollama', 'ps'], 
							  stdout=subprocess.PIPE, 
							  stderr=subprocess.PIPE, 
							  text=True,
							  check=True)
		
		# 分割输出行并移除表头
		lines = result.stdout.strip().split('\n')[1:]
		
		# 如果只有表头或空行，说明没有运行中的模型，需要启动
		if not lines or not lines[0].strip():
			return True
			
		# 解析运行中的模型
		running_models = []
		for line in lines:
			if line.strip():
				model_name = line.split()[0]
				running_models.append(model_name)
		
		# 如果有多个模型或单个不匹配的模型，停止所有模型后递归检查
		if len(running_models) > 1 or (running_models and running_models[0] != target_model):
			print("检测到其他模型在运行，正在停止...")
			for model in running_models:
				stop_model(model)
			return check_running_models(target_model)
			
		# 如果只有一个模型且是目标模型，不需要启动
		if running_models and running_models[0] == target_model:
			print(f"模型 {target_model} 已经在运行")
			return False
			
	except subprocess.CalledProcessError as e:
		print(f"检查运行中模型失败: {e.stderr}")
		raise

def start_model(model_name):
	"""启动模型"""
	try:
		print(f"正在启动模型 {model_name}...")
		startupinfo = None
		if os.name == 'nt':  # Windows系统
			startupinfo = subprocess.STARTUPINFO()
			
		process = subprocess.Popen(
			['ollama', 'run', model_name],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			encoding='utf-8',  # 显式指定UTF-8编码
			
		)
		
		# 等待一小段时间确保模型启动
		time.sleep(3)
		
		# 检查进程是否正常启动
		if process.poll() is not None and process.returncode != 0:
			
			stdout, stderr = process.communicate()
			print(f"进程输出: {stdout}")
			print(f"错误输出: {stderr}")
			raise subprocess.CalledProcessError(
				process.returncode, 
				['ollama', 'run', model_name], 
				stdout, 
				stderr
			)
			
		print(f"模型 {model_name} 已启动")
		return process  # 返回进程对象以便后续管理
		
	except subprocess.CalledProcessError as e:
		print(f"启动模型失败: {e.stderr}")
		raise
	except Exception as e:
		print(f"启动模型时发生异常: {str(e)}")
		raise


def stop_model(model_name):
	"""停止模型"""
	try:
		print(f"正在停止模型 {model_name}...")
		startupinfo = None
		if os.name == 'nt':
			startupinfo = subprocess.STARTUPINFO()
			startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
			
		result = subprocess.run(
			['ollama', 'stop', model_name],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			encoding='utf-8',
			errors='ignore',
			check=True,
			startupinfo=startupinfo
		)
		print(f"模型 {model_name} 已停止")
	except subprocess.CalledProcessError as e:
		print(f"停止模型失败: {e.stderr}")


def signal_handler(signum, frame):
	"""处理中断信号"""
	print("\n接收到中断信号，正在清理...")
	# 停止模型
	stop_model(config["server"]["model"])
	# 退出程序
	exit(0)


def cleanup():
	"""退出时的清理工作"""
	try:
		print("\n正在执行清理工作...")
		stop_model(config["server"]["model"])
	except Exception as e:
		print(f"清理时发生错误: {e}")


if __name__ == "__main__":
	# 注册信号处理器
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)
	
	model_name = config["server"]["model"]
	
	# 检查并启动模型
	if check_running_models(model_name):
		start_model(model_name)
	
	# 注册退出处理器
	atexit.register(cleanup)
	
	try:
		usage_q = queue.Queue()
		output_dir = os.path.join("eval_results", re.sub(r"\W", "-", model_name))
		os.makedirs(output_dir, exist_ok=True)
		log_path = os.path.join(output_dir, "report.txt")
		try:
			os.remove(log_path)
		except:
			pass
		config_copy = copy.deepcopy(config)
		del config_copy["server"]["api_key"]
		del config_copy["test"]["categories"]
		log(f"{datetime.now()}")
		log(json.dumps(config_copy, indent="\t"))
		assigned_subjects = config["test"]["categories"]
		start = time.time()
		evaluate(assigned_subjects)
		end = time.time()
		log(f"Finished the benchmark in {elapsed(start)}.")
		final_report(assigned_subjects)
		print("Report saved to:", log_path)
	except KeyboardInterrupt:
		print("\n程序被用户中断")
		raise
