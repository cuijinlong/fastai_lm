import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成200个设备样本
n_samples = 200

# 创建数据集 - 使用中文列名
data = {
    '设备ID': [f'EQP_{i:03d}' for i in range(1, n_samples + 1)],

    # 设备基本信息
    '设备类型': np.random.choice(['风机', '泵机', '压缩机', '发电机', '传送带'], n_samples,
                                 p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    '安装年限': np.random.normal(5, 2, n_samples).astype(int),
    '设备负载': np.random.choice(['低负载', '正常负载', '高负载'], n_samples, p=[0.2, 0.6, 0.2]),
    '工作环境': np.random.choice(['清洁', '一般', '恶劣'], n_samples, p=[0.3, 0.5, 0.2]),

    # 运行参数特征
    '平均温度_摄氏度': np.random.normal(75, 15, n_samples),
    '最大温度_摄氏度': np.random.normal(85, 20, n_samples),
    '平均振动_毫米秒': np.random.uniform(2, 15, n_samples),
    '峰值振动_毫米秒': np.random.uniform(3, 25, n_samples),
    '噪声水平_db': np.random.normal(80, 10, n_samples),
    '电流波动_百分比': np.random.uniform(1, 20, n_samples),

    # 维护历史
    '累计运行小时': np.random.uniform(1000, 20000, n_samples),
    '上次维护至今小时': np.random.uniform(100, 2000, n_samples),
    '历史维修次数': np.random.poisson(3, n_samples),
    '预防性维护评分': np.random.uniform(1, 10, n_samples),

    # 故障相关特征
    '轴承状态': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),  # 0=正常,1=预警,2=异常
    '润滑状态': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
    '密封件状态': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    '对中精度_毫米': np.random.uniform(0.01, 0.5, n_samples),

    # 预测指标（用于回归分析）
    '剩余寿命预测_天': np.random.uniform(30, 365, n_samples),
    '维修成本预测_千元': np.random.uniform(5, 100, n_samples),

    # 生存分析相关
    '监测时间_天': np.random.weibull(2, n_samples) * 180,
    '故障事件': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    '故障时间_天': np.full(n_samples, np.nan)
}

# 创建DataFrame
equipment_dataset = pd.DataFrame(data)

# 对于发生故障的设备，设置故障时间（在监测期内）
failure_mask = equipment_dataset['故障事件'] == 1
equipment_dataset.loc[failure_mask, '故障时间_天'] = np.random.uniform(
    30,
    equipment_dataset.loc[failure_mask, '监测时间_天'],
    failure_mask.sum()
)

# 添加相关性以模拟真实数据模式
# 高振动与较短剩余寿命相关
high_vibration_mask = equipment_dataset['峰值振动_毫米秒'] > 15
equipment_dataset.loc[high_vibration_mask, '剩余寿命预测_天'] *= 0.6
equipment_dataset.loc[high_vibration_mask, '故障事件'] = np.random.choice([0, 1], high_vibration_mask.sum(),
                                                                          p=[0.5, 0.5])

# 恶劣环境与较高故障率相关
harsh_env_mask = equipment_dataset['工作环境'] == '恶劣'
equipment_dataset.loc[harsh_env_mask, '剩余寿命预测_天'] *= 0.7
equipment_dataset.loc[harsh_env_mask, '故障事件'] = np.random.choice([0, 1], harsh_env_mask.sum(), p=[0.4, 0.6])

# 高负载与较高维修成本相关
high_load_mask = equipment_dataset['设备负载'] == '高负载'
equipment_dataset.loc[high_load_mask, '维修成本预测_千元'] *= 1.5

# 清理数据，确保数值在合理范围内
equipment_dataset['安装年限'] = np.clip(equipment_dataset['安装年限'], 1, 15)
equipment_dataset['平均温度_摄氏度'] = np.clip(equipment_dataset['平均温度_摄氏度'], 20, 120)
equipment_dataset['峰值振动_毫米秒'] = np.clip(equipment_dataset['峰值振动_毫米秒'], 2, 30)
equipment_dataset['剩余寿命预测_天'] = np.clip(equipment_dataset['剩余寿命预测_天'], 10, 400)
equipment_dataset['维修成本预测_千元'] = np.clip(equipment_dataset['维修成本预测_千元'], 2, 150)

# 显示数据集信息
print("大型机械设备故障监测数据集概况:")
print(f"样本数量: {len(equipment_dataset)}")
print(f"特征数量: {len(equipment_dataset.columns)}")
print("\n前5行数据:")
print(equipment_dataset.head())

# 将数据写入Excel文件
excel_filename = "大型机械设备故障监测数据集.xlsx"

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 写入主数据集
    equipment_dataset.to_excel(writer, sheet_name='设备监测数据', index=False)

    # 创建数据字典表
    data_dictionary = {
        '变量名称': [
            '设备ID', '设备类型', '安装年限', '设备负载', '工作环境',
            '平均温度_摄氏度', '最大温度_摄氏度', '平均振动_毫米秒', '峰值振动_毫米秒', '噪声水平_db',
            '电流波动_百分比', '累计运行小时', '上次维护至今小时', '历史维修次数', '预防性维护评分',
            '轴承状态', '润滑状态', '密封件状态', '对中精度_毫米',
            '剩余寿命预测_天', '维修成本预测_千元', '监测时间_天', '故障事件', '故障时间_天'
        ],
        '变量说明': [
            '设备唯一标识符',
            '设备类型分类',
            '设备安装使用年限（年）',
            '设备运行负载状态',
            '设备工作环境条件',
            '设备运行时平均温度',
            '设备运行时达到的最高温度',
            '设备平均振动水平',
            '设备峰值振动水平',
            '设备运行噪声水平',
            '电流波动幅度百分比',
            '设备累计运行时间',
            '距离上次维护的时间',
            '设备历史维修总次数',
            '预防性维护执行情况评分（1-10分）',
            '轴承健康状态（0=正常，1=预警，2=异常）',
            '润滑系统状态（0=正常，1=预警，2=异常）',
            '密封件状态（0=正常，1=异常）',
            '设备对中精度偏差',
            '预测的设备剩余使用寿命',
            '预测的下次维修成本',
            '总监测时间',
            '故障发生事件（0=正常，1=故障）',
            '故障发生时间（天，仅故障设备有数据）'
        ],
        '分析用途': [
            '标识', '分类', '风险因素', '运行状态', '环境因素',
            '状态监测', '异常检测', '状态监测', '故障预测', '状态监测',
            '电气健康', '寿命分析', '维护计划', '历史分析', '维护质量',
            '部件健康', '润滑系统', '密封系统', '机械精度',
            '回归目标', '回归目标', '生存分析', '分类目标', '生存分析'
        ]
    }

    data_dict_df = pd.DataFrame(data_dictionary)
    data_dict_df.to_excel(writer, sheet_name='数据字典', index=False)

print(f"\n数据集已成功写入Excel文件: {excel_filename}")