# 一个简单的非工业化的污水处理厂模型API

这个仓库里的代码用Flask框架部署一个简单污水处理厂的深度学习模型API。  

API接受一个tsv格式的文件，文件输入为5天的污水处理厂数据，预测五天后的总氮数据。  

以下代码为在自己电脑端启动flask server   
 
```sh
$ python run_keras_server.py 
Using TensorFlow backend.
 * Loading Keras model and Flask starting server...please wait until server has fully started
...
 * Running on http://127.0.0.1:5000
```  

以下代码为仿造一个上传文件的请求，以json格式返回结果并展示预测结果  

```sh
$ python simple_request.py 
```   

