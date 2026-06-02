<!--
 * @Author: kevincnzhengyang kevin.cn.zhengyang@gmail.com
 * @Date: 2026-06-02 10:18:47
 * @LastEditors: kevincnzhengyang kevin.cn.zhengyang@gmail.com
 * @LastEditTime: 2026-06-02 10:36:08
 * @FilePath: /qlib_data/architecture.md
 * @Description: 
 * 
 * Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
-->
# 目标
将QLib的data layer层抽取为独立的项目

# 文件结构
目前的路径是～/Quanter/aoc/qlib_data
- reference QLib的相关源码
- qlib_data 工作目录，成果源码。
- tests 测试代码 

# 接口
- init 初始化接口，来自qlib.init
- load_dataset 读取接口，来自qlib.data.D.features
- dump_dataset 写入接口，来自qlib.scripts.dump_bin.dump_all

# 工作约束
- 分析reference/qlib下的代码
- 完成代码移植和接口封装，不能依赖qlib或者pyqlib库
- 移除不需要的第三方包依赖
- 编写新的接口对原有内容进行封装
- 按照新的源代码结构，修改引用关系
- 编写测试代码进行测试，测试数据使用tests/HK.00100.csv文件作为数据源
- 最后将必须依赖的第三方包，输出为requirements.txt