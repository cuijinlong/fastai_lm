import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成200个样本
n_samples = 200

# 创建数据集 - 使用中文列名
data = {
    '患者ID': [f'BCC_{i:03d}' for i in range(1, n_samples + 1)],

    # 人口统计学特征
    '年龄': np.random.normal(65, 12, n_samples).astype(int),
    '性别': np.random.choice(['男性', '女性'], n_samples, p=[0.55, 0.45]),
    '皮肤类型': np.random.choice(['I型', 'II型', 'III型', 'IV型'], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
    '家族史': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),

    # 临床特征
    '肿瘤大小_毫米': np.random.uniform(2, 40, n_samples),
    '肿瘤位置': np.random.choice(['面部', '头皮', '颈部', '躯干', '四肢'], n_samples, p=[0.5, 0.1, 0.15, 0.15, 0.1]),
    '溃疡形成': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    '色素沉着': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    '边界不规则性': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),

    # 组织病理学特征
    '组织学亚型': np.random.choice(['结节型', '浅表型', '浸润型', '微结节型'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    '神经侵犯': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    '切缘阴性': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    '肿瘤深度_毫米': np.random.uniform(0.5, 8, n_samples),
    '核分裂计数': np.random.poisson(3, n_samples),

    # 风险因素
    '日晒暴露': np.random.choice(['轻度', '中度', '重度'], n_samples, p=[0.3, 0.5, 0.2]),
    '既往BCC病史': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    '免疫抑制': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    '吸烟状态': np.random.choice(['从不吸烟', '已戒烟', '当前吸烟'], n_samples, p=[0.4, 0.4, 0.2]),

    # 治疗相关特征
    '治疗方式': np.random.choice(['手术切除', '莫氏手术', '刮除术', '冷冻治疗', '放射治疗'], n_samples,
                                 p=[0.4, 0.3, 0.15, 0.1, 0.05]),
    '治疗成功': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),

    # 预后指标（用于回归分析）
    '复发风险评分': np.random.uniform(1, 10, n_samples),
    '愈合时间_天': np.random.gamma(20, 1, n_samples).astype(int),

    # 生存分析相关
    '随访时间_月': np.random.weibull(2, n_samples) * 36,
    '复发事件': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    '复发时间_月': np.full(n_samples, np.nan)
}

# 创建DataFrame
bcc_dataset = pd.DataFrame(data)

# 对于发生复发的患者，设置复发时间（在随访期内）
recurrence_mask = bcc_dataset['复发事件'] == 1
bcc_dataset.loc[recurrence_mask, '复发时间_月'] = np.random.uniform(
    6,
    bcc_dataset.loc[recurrence_mask, '随访时间_月'],
    recurrence_mask.sum()
)

# 添加一些相关性以模拟真实数据模式
# 较大的肿瘤尺寸与较高的复发风险相关
bcc_dataset.loc[bcc_dataset['肿瘤大小_毫米'] > 20, '复发风险评分'] *= 1.5
bcc_dataset.loc[bcc_dataset['肿瘤大小_毫米'] > 20, '复发事件'] = np.random.choice([0, 1], (
            bcc_dataset['肿瘤大小_毫米'] > 20).sum(), p=[0.6, 0.4])

# 浸润性亚型与较高复发风险相关
infiltrative_mask = bcc_dataset['组织学亚型'] == '浸润型'
bcc_dataset.loc[infiltrative_mask, '复发风险评分'] *= 1.8
bcc_dataset.loc[infiltrative_mask, '复发事件'] = np.random.choice([0, 1], infiltrative_mask.sum(), p=[0.5, 0.5])

# 神经侵犯与较差预后相关
pni_mask = bcc_dataset['神经侵犯'] == 1
bcc_dataset.loc[pni_mask, '复发风险评分'] *= 2.0
bcc_dataset.loc[pni_mask, '复发事件'] = np.random.choice([0, 1], pni_mask.sum(), p=[0.3, 0.7])

# 清理数据，确保数值在合理范围内
bcc_dataset['年龄'] = np.clip(bcc_dataset['年龄'], 30, 95)
bcc_dataset['肿瘤大小_毫米'] = np.clip(bcc_dataset['肿瘤大小_毫米'], 1, 50)
bcc_dataset['肿瘤深度_毫米'] = np.clip(bcc_dataset['肿瘤深度_毫米'], 0.1, 12)
bcc_dataset['核分裂计数'] = np.clip(bcc_dataset['核分裂计数'], 0, 15)
bcc_dataset['复发风险评分'] = np.clip(bcc_dataset['复发风险评分'], 1, 15)
bcc_dataset['愈合时间_天'] = np.clip(bcc_dataset['愈合时间_天'], 7, 120)

# 显示数据集信息
print("基底细胞癌研究数据集概况:")
print(f"样本数量: {len(bcc_dataset)}")
print(f"特征数量: {len(bcc_dataset.columns)}")
print("\n前5行数据:")
print(bcc_dataset.head())

print("\n数据集统计摘要:")
print(bcc_dataset.describe())

print("\n分类变量分布:")
categorical_cols = ['性别', '皮肤类型', '肿瘤位置', '组织学亚型',
                    '日晒暴露', '吸烟状态', '治疗方式', '复发事件']
for col in categorical_cols:
    print(f"\n{col}:")
    print(bcc_dataset[col].value_counts())

# 将数据写入Excel文件
excel_filename = "基底细胞癌研究数据集.xlsx"

# 创建一个Excel写入器
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 写入主数据集
    bcc_dataset.to_excel(writer, sheet_name='基底细胞癌数据', index=False)

    # 创建数据字典表
    data_dictionary = {
        '变量名称': [
            '患者ID', '年龄', '性别', '皮肤类型', '家族史',
            '肿瘤大小_毫米', '肿瘤位置', '溃疡形成', '色素沉着', '边界不规则性',
            '组织学亚型', '神经侵犯', '切缘阴性', '肿瘤深度_毫米', '核分裂计数',
            '日晒暴露', '既往BCC病史', '免疫抑制', '吸烟状态',
            '治疗方式', '治疗成功', '复发风险评分', '愈合时间_天',
            '随访时间_月', '复发事件', '复发时间_月'
        ],
        '变量说明': [
            '患者唯一标识符',
            '患者年龄（岁）',
            '患者性别',
            '皮肤类型（Fitzpatrick分型：I-VI）',
            '皮肤癌家族史（0=无，1=有）',
            '肿瘤最大直径（毫米）',
            '肿瘤解剖位置',
            '溃疡形成（0=无，1=有）',
            '色素沉着（0=无，1=有）',
            '边界不规则性（0=规则，1=轻度不规则，2=明显不规则）',
            '组织学亚型',
            '神经侵犯（0=无，1=有）',
            '手术切缘阴性（0=阳性，1=阴性）',
            '肿瘤浸润深度（毫米）',
            '核分裂计数（每高倍视野）',
            '日晒暴露程度',
            '既往基底细胞癌病史（0=无，1=有）',
            '免疫抑制状态（0=无，1=有）',
            '吸烟状态',
            '治疗方式',
            '治疗成功（0=失败，1=成功）',
            '复发风险评分（1-15分，分数越高风险越大）',
            '创面愈合时间（天）',
            '总随访时间（月）',
            '复发事件（0=未复发，1=复发）',
            '复发时间（月，仅复发患者有数据）'
        ],
        '数据类型': [
            '字符串', '整数', '分类', '分类', '二分类',
            '连续', '分类', '二分类', '二分类', '有序分类',
            '分类', '二分类', '二分类', '连续', '整数',
            '分类', '二分类', '二分类', '分类',
            '分类', '二分类', '连续', '整数',
            '连续', '二分类', '连续'
        ],
        '分析用途': [
            '标识', '回归/分类', '分类', '风险因素', '风险因素',
            '回归/分类', '分类', '预后因素', '临床特征', '临床特征',
            '预后因素', '预后因素', '治疗结果', '预后因素', '预后因素',
            '风险因素', '风险因素', '风险因素', '风险因素',
            '治疗', '分类', '回归目标', '回归目标',
            '生存分析', '生存分析', '生存分析'
        ]
    }

    data_dict_df = pd.DataFrame(data_dictionary)
    data_dict_df.to_excel(writer, sheet_name='数据字典', index=False)

    # 创建统计分析表
    stats_summary = bcc_dataset.describe(include='all').transpose()
    stats_summary.to_excel(writer, sheet_name='统计摘要')

print(f"\n数据集已成功写入Excel文件: {excel_filename}")
print("文件包含三个工作表:")
print("1. 基底细胞癌数据 - 主要数据集")
print("2. 数据字典 - 数据字典和变量说明")
print("3. 统计摘要 - 统计分析摘要")