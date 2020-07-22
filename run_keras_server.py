# import the necessary packages
import keras
from keras.models import load_model
import numpy as np
import flask
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, redirect, url_for


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

config = tf.compat.v1.ConfigProto(
		intra_op_parallelism_threads=1,
		allow_soft_placement=True
	)
session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)


def web_load_model():
	# load the trained model for TNe-hour
	global model
	path = 'TNe-hour.h5'
	model = load_model(path)

	# for solving the problem when load_model and model_predict through
	# web is not in the same thread
	testdata = np.zeros(shape=(1,5,8))
	pred = model.predict(testdata)
	#print(model.input_shape)


def prepare_data(data, target):
	# resize the input data and preprocess it
	# the input data should be 5 x 8
	# 5 days of 'volume', 'CODi','NH3-Ni', 'TNi', 'TPi', 'CODe', 'NH3Ne', 'TPe'
	# todo: preprocessing
	data = data.reshape(target)
	data = np.expand_dims(data, axis=0)
	# print(data.shape)
	# return the processed data
	return data


@app.route('/')
def hello():
	return render_template("hello.html")


@app.route("/predict", methods=["GET","POST"])
def predict():
	# initialize the result dictionary that will be returned from the
	# view
	result = {"success": False}

	# ensure the data was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files['data']:
			# read the data in tsv format
			data = pd.read_csv(flask.request.files["data"], sep='\t', header=None)
			data = np.array(data)
			# preprocess the data and prepare it for prediction
			data = prepare_data(data, target=(5, 8))

			try:
				with session.as_default():
					with session.graph.as_default():
						# classify the input data and then initialize the list
						# of predictions to return to the client
						preds = model.predict(data)
						result["TNe"] = str(preds.flatten()[0])

						# indicate that the request was a success
						result["success"] = True
			except Exception as ex:
				print('Seatbelt Prediction Error', ex)
		# return the data dictionary as a JSON response
		return render_template('result.html', result=result)
	elif flask.request.method == "GET":
		return render_template('predict.html')

@app.route("/about_us")
def about_us():
	info = [{"name":"廉勍",
			 "desc":"环境科学与工程学院2018级环境工程研究生。研究课题为“基于机器学习的污水处理过程模型预测控制”研究。目前以江苏省一座处理规模18万吨/天的污水处理厂为研究对象，开展机器学习模型、污水处理机理模型和CFD流程模拟工作。",
			 "img":'lianqing.png'},
			{"name": "向远坤",
			 "desc": "同济大学环境学院2017级环境工程本科生。同济大学三等奖学金。有较强的空间想象和图形转换能力。在本次项目中负责污水厂运行CFD流场模型构建和模拟运行工作。",
			 "img":'xiangyuankun.png'},
			{"name": "姬永顺",
			 "desc": "同济大学经济与管理学院2017级金融学本科生。负责的UNISKARE项目曾获哥伦比亚大学模拟企业家商赛全国季军，互联网+校内银奖，并入驻同济创业谷孵化，商赛经历丰富。本次项目负责制定财务预算、公司管理规划，参与市场分析和风险控制。",
			 "img":'jiyongshun.jpg'},
			{"name": "司雨萱",
			 "desc": "同济大学经济与管理学院2017级工程管理专业本科生。曾获同济大学优秀奖学金一等奖，同济大学第九届应力创新大赛一等奖，第二届厉兵秣马创新创业训练赛一等奖，中国启东创新创业大赛三等奖，同济大学数学建模三等奖，在本项目中负责商业策划部分。",
			 "img":'siyuxuan.jpg'},
			{"name": "尹晓庆",
			 "desc": "同济大学环境科学与工程学院2018级环境工程本科生。掌握C#、python。曾获国家奖学金、社会活动奖学金、上海市计算机应用能力大赛二等奖、同济大学数学建模竞赛一等奖。在本次项目中负责智能模型的开发。",
			 "img":'yinxiaoqing.png'},
			{"name": "谢一凡",
			 "desc": "同济大学环境科学与工程学院2018级给排水科学与工程本科生。曾参与关于污水氧化高级处理课题，曾获全国大学生数学竞赛三等奖、同济大学数学建模竞赛三等奖、“中青杯”建模比赛优胜奖，在本次项目中负责智能模型组的构建与开发部分工作。",
			 "img":'xieyifan.jpg'},
			{"name": "陈玮豪",
			 "desc": "同济大学环境科学与工程学院2018级给排水科学与工程本科生。曾获上海市计算机应用能力大赛二等奖，同济大学数学建模竞赛一等奖，所在团队申请到上海市创新创业项目一项。曾获冠龙奖学金。在本次项目中负责参与智能模型平台架构和开发。",
			 "img":'chenweihao.png'},
			{"name": "鲁思琪",
			 "desc": "同济大学环境科学与工程学院2018级排水科学与工程本科生。曾获第一届水环境与水生态科普创意大赛视频类单项奖（团队），第二届水环境与水生态科普创意大赛入围新媒体类决赛（团队）；于本项目中主要负责数字孪生系统机理内核搭建。",
			 "img":'lusiqi.png'},
			{"name": "邓博苑",
			 "desc": "同济大学环境科学与工程学院2018级本科生。掌握污水处理相关机理模型等知识，曾参与超润湿网膜的功能化设计及其对污水快速油水分离的课题研究，曾获同济大学二等奖学金，全国计算机等级考试二级证书。在本次项目中负责机理模型的构建。",
			 "img":'dengboyuan.jpg'},
			{"name": "陈咏琪",
			 "desc": "同济大学环境科学与工程学院2018级环境工程本科生。掌握Ps、Pr、3ds Max等多媒体技能。参与同济大学SITP大学生创新创业训练计划项目“土壤重金属污染预测模型的建立与应用”。在本次项目中负责数字孪生系统机理内核的搭建",
			 "img":''},
			{"name": "朱海龙",
			 "desc": "同济大学环境科学与工程学院2018级给排水工程本科生。曾获同济大学给水排水88级“一滴水”校友奖学金，肯特杨钦环境教育奖励金-优秀干部奖。在本次项目中负责将处理工艺机理与人工智能模型进行有机结合与架构。",
			 "img":''},
			{"name": "常天玺",
			 "desc": "同济大学环境科学与工程学院2018级环境工程本科生。曾参与微生物代谢功能调控策略与强化生物脱氮工艺及机理研究等项目。在本次项目中将负责处理工艺机理与人工智能模型有机结合与架构。",
			 "img":'changtianxi.png'},
			{"name": "祁好雨",
			 "desc": "同济大学电子与信息工程学院2018级计算机科学与技术系本科生。曾获同济大学优秀学生一等奖学金，国家奖学金。在本次项目中负责神经网络模型的构建和调优，学习模型的封装以及网站的编写。",
			 "img":'qihaoyu.png'},
			{"name": "田沁锋",
			 "desc": "同济大学设计创意学院2017级工业设计专业本科生。本科期间完成涉及家具、医疗器械、宣传营销等多个设计课题，曾获同济大学一等奖学金、上海市工业设计大赛二、三等奖，在本次项目中主要负责建模表达工作。",
			 "img":'tianqinfeng.jpg'
			}]
	info[0]['name'] = '项目负责人：'+info[0]['name']
	for person in info:
		person['img'] = '/avatar/'+person['img']
	for person in info[1:]:
		person['name'] = '团队成员：'+person['name']
	info1 = info[:3]
	info2 = info[3:6]
	info3 = info[6:9]
	info4 = info[9:12]
	info5 = info[12:]
	return render_template("about_us.html",info1=info1,info2=info2,info3=info3,info4=info4,info5=info5)


@app.route("/about_program")
def about_program():
	return render_template("about_program.html")


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	web_load_model()
	# app.run(host='0.0.0.0',port=5000)
	app.run()