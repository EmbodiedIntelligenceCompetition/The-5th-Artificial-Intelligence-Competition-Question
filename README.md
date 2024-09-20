## 数据集

我们采用iGIbson的公开场景数据集，不提供训练使用的episode数据集。用于评测的episode数据集对选手保密。具体下载方式请参考：

https://stanfordvl.github.io/iGibson/dataset.html

选手需要下载dataset和assets。






## 参赛指南

### 项目配置

* **步骤1**：安装anaconda并且创建一个环境：

  ```
  conda create -n gibson_2024 python=3.7
  conda activate gibson_2024
  ```

* **步骤2**：安装EGL依赖：

  ```
  sudo apt-get install libegl1-mesa-dev
  ```

* **步骤3**：安装iGibson

  ```bash
  git clone https://github.com/StanfordVL/iGibson.git --recursive
  cd iGibson
  pip install -e .
  ```

* **步骤4**：安装baseline：

  ```
  pip install -r requirements.txt
  pip install -e .
  ```
  如果安装遇到问题，可以尝试更新pip版本: pip instal --upgrade pip
  
* **步骤5**：配置os.environ['HOME]：

  ```
  进入agent/utils/common.py，设置os.environ['HOME']为你的home目录
  # 示例
  import os
  # 设置环境变量 HOME
  os.environ["HOME"] = "/path/to/your/home/directory"
  ```
  
* **步骤6**：推荐使用cuda11.7版本。




### 训练

#### 配置文件

有agent的配置文件和igibson的配置文件两部分。agent的配置文件位于agent/configs下，igibson的配置文件位于agent/gibson_extension/examples/configs下

#### 使用docker



#### 不使用docker

* **步骤1**：进入training文件夹：

  ```
  cd agent/training
  ```

* **步骤2**：启动训练：

  ```
  bash point_nav_ppo_train.sh
  ```

  这个启动了两个环境进行训练，在单卡3090上约5天完成训练

### 本地评测

* **步骤1**：进入training文件夹：

  ```
  cd agent/training
  ```

* **步骤2**：随机生成一些场景的episode data：

  ```
  bash generate_data.sh
  ```
  这个将在agent/training目录下生成测试数据的json文件

* **步骤3**：启动检查点的评估：

  ```
  bash point_nav_ppo_eval.sh
  ```
  需要在igibson的配置文件中修改scene_episode_config_name属性，使其为步骤2生成的json文件所在的文件夹



### 在线评测


