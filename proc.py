# 创建一个变量来记录当前的标签
current_label = None
# 创建一个变量来记录当前打开的输出文件
current_file = None

# 遍历输入文件，一行一行地读取
with open("all.tsv", "r") as f:
    for line in f:
        label = int(line.split("\t")[-1].strip()) # 获取最后一列的值作为标签
        if label != current_label: # 如果标签发生了变化
            if current_file: # 如果有打开的输出文件，关闭它
                current_file.close()
            current_label = label # 更新当前标签
            if label>= 300 and label <400:
                current_file = open("raw/attack/"+ str(label) + ".tsv", "w") # 打开一个新的输出文件
            else:
                current_file = open("raw/benign/"+ str(label) + ".tsv", "w") # 打开一个新的输出文件
        current_file.write(line) # 将行写入当前输出文件

# 最后关闭最后一个打开的输出文件
if current_file:
    current_file.close()