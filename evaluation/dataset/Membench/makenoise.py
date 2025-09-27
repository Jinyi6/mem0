import json
import numpy as np
import random
import argparse


def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path,'w', encoding='utf-8') as f:
        json.dump(data,f, indent=4,ensure_ascii=False)

def infuse_single_trajectory_message(traj, noise_pool, length):
    if length == 0:
        noisy_traj = {}
        noisy_traj['tid'] = traj['tid']
        noisy_traj['message_list'] =  ['{} (place: {}; time{})'.format(m['message'], m['place'], m['time']) for m in traj['message_list']]
        noisy_traj['QA'] = traj['QA']
        return noisy_traj
    
    noisy_traj = {}
    noisy_traj['tid'] = traj['tid']
    new_message_list = []
    raw_message_list = traj['message_list']

    total_step = len(raw_message_list) + length  # 一个noise message长度为3

    # print(len(raw_message_list))
    tmp = sorted(np.random.choice(range(total_step), size=len(raw_message_list), replace=False))
    # print(len(tmp))
    relocate_dict = {(int(tmp) - index) * 3 + index : index for index, tmp in enumerate(tmp)}
    reverse_relocate_dict = {index: (int(tmp) - index) * 3 + index for index, tmp in enumerate(tmp)}
    # print(relocate_dict)
    # reverse_relocate_dict = {index:int(tmp) for index, tmp in enumerate(tmp)}  ## 方便获取新的target_step_id
    
    noise_list = np.random.choice(noise_pool, size=length, replace=False)
    noise_id = 0
    index = 0
    
    while index < len(raw_message_list) + (3 * length) - 1:
        # print(index)
        if index in relocate_dict:
            # print(index)
            re_index = relocate_dict[index]
            new_message_list.append('{} (place: {}; time: {})'.format(raw_message_list[re_index]['message'], raw_message_list[re_index]['place'], raw_message_list[re_index]['time']))
        else:
            
            noise_message = ['{}'.format(i['message']) for i in noise_list[noise_id]['noise_message']]
            new_message_list.extend(noise_message)
            noise_id += 1
            
            index += 2 ## 连续跳过4个
            # print(index)
        index += 1
    
    noisy_traj['message_list'] = new_message_list
    noisy_traj['QA'] = traj['QA']
    noisy_traj['QA']['target_step_id'] = [reverse_relocate_dict[step_id] for step_id in noisy_traj['QA']['target_step_id']]

    return noisy_traj

def infuse_single_trajectory_session(traj, noise_pool, length):
    if length == 0:
        noisy_traj = {}
        noisy_traj['tid'] = traj['tid'] 
        new_message_list = []
        for m in traj['message_list']:
            # print(m)
            for m_i in m:
                try:
                    new_message_list.append({
                        'user': '{} (place: {}; time: {})'.format(m_i['user_message'], m_i['place'], m_i['time']),
                        'agent': '{} (place: {}; time: {})'.format(m_i['assistant_message'], m_i['place'], m_i['time'])
                    })
                except:
                    new_message_list.append({
                        'user': '{} (place: {}; time: {})'.format(m_i['user'], m_i['place'], m_i['time']),
                        'agent': '{} (place: {}; time: {})'.format(m_i['assistant'], m_i['place'], m_i['time'])
                    })
        noisy_traj['message_list'] =  new_message_list
        noisy_traj['QA'] = traj['QA']
        noisy_traj['QA']['target_step_id'] = [id[0] for id in noisy_traj['QA']['target_step_id']]

        return noisy_traj
    
    noisy_traj = {}
    noisy_traj['tid'] = traj['tid']
    new_message_list = []
    raw_message_list = traj['message_list'] ## 这里格式为[[], [], []]

    total_traj = len(raw_message_list) + length

    tmp = sorted(np.random.choice(range(total_traj), size=len(raw_message_list), replace=False))
    relocate_dict = {int(tmp): index for index, tmp in enumerate(tmp)}
    reverse_relocate_dict = {index: int(tmp)-index for index, tmp in enumerate(tmp)}

    noise_list = np.random.choice(noise_pool, size=length, replace=False)
    noise_id = 0
    re_index = 0
    for index in range(total_traj):
        if index in tmp:
            # print(index)
            for i in raw_message_list[re_index]:
                # print(i)
                try: 
                    new_message_list.append({
                        'user': '{} (place: {}; time: {})'.format(i['user_message'], i['place'], i['time']),
                        'agent': '{} (place: {}; time: {})'.format(i['assistant_message'], i['place'], i['time'])
                    })
                except:
                    new_message_list.append({
                        'user': '{} (place: {}; time: {})'.format(i['user'], i['place'], i['time']),
                        'agent': '{} (place: {}; time: {})'.format(i['assistant'], i['place'], i['time'])
                    })
            re_index += 1
        else:
            noise_message = [{
                'user': '{}'.format(i['user']),
                'agent': '{}'.format(i['assistant'])
            } for i in noise_list[noise_id]['noise_message']]
            new_message_list.extend(noise_message)
            noise_id += 1
    
    noisy_traj['message_list'] = new_message_list
    noisy_traj['QA'] = traj['QA']
    
    noisy_traj['QA']['target_step_id'] = [reverse_relocate_dict[step_id[1]] * 3 + step_id[0] for step_id in noisy_traj['QA']['target_step_id']]

    return noisy_traj

def MakeNoiseSession(RawDataPath, Outdir, length, sample_num=100):
    NoiseDataPath = 'NoiseData/FirstNoise.json'
    NoisePool = load_json(NoiseDataPath)

    output_data = {}

    RawData = load_json(RawDataPath)
    for QAtype, QAtype_data in RawData.items():  # 获得simple
        output_data[QAtype] = {}
        for scenario, scenario_data in QAtype_data.items():  # 获得role
            output_data[QAtype][scenario] = []
            scenario_traj_all_pre = []
            for traj in scenario_data:
                noisy_traj = infuse_single_trajectory_session(traj, NoisePool, length)
                # output_data[QAtype][scenario].append(noisy_traj)
                scenario_traj_all_pre.append(noisy_traj)
          
            scenario_traj_all = random.sample(scenario_traj_all_pre, sample_num)
            output_data[QAtype][scenario] = scenario_traj_all
            print(QAtype, scenario, 'Finish!')
    
    RawDataPathSplit = RawDataPath.split('/')
    output_path = Outdir + '/' + RawDataPathSplit[1].replace('.json', '') + '_multiple_{}.json'.format(length)
    save_json(output_data, output_path) 

def infuse_single_trajectory_message_highlevel(traj, noise_pool, length):
    # print(len(traj['message_list']))
    if length == 0:
        noisy_traj = {}
        noisy_traj['tid'] = traj['tid']
        noisy_traj['message_list'] =  ['{} (place: {}; time: {})'.format(m['message'], m['place'], m['time']) for m in traj['message_list'][0]]
        noisy_traj['QA'] = traj['QA']
        noisy_traj['QA']['target_step_id'] = [id[0] for id in noisy_traj['QA']['target_step_id']]

        return noisy_traj
    
    noisy_traj = {}
    try:
        noisy_traj['tid'] = traj['tid']
    except:
        noisy_traj['tid'] = traj['gid']
        
    new_message_list = []
    raw_message_list = traj['message_list'] ## 这里格式为[[], [], []]

    total_traj = len(raw_message_list) + length

    tmp = sorted(np.random.choice(range(total_traj), size=len(raw_message_list), replace=False))
    relocate_dict = {int(tmp): index for index, tmp in enumerate(tmp)}
    reverse_relocate_dict = {index: int(tmp)-index for index, tmp in enumerate(tmp)}

    noise_list = np.random.choice(noise_pool, size=length, replace=False)
    noise_id = 0
    re_index = 0
    # print(tmp)
    for index in range(total_traj):
        if index in tmp:
            # print(raw_message_list)
            for i in raw_message_list[re_index]:
                # print(i)
                new_message_list.append('{} (place: {}; time: {})'.format(i['message'], i['place'], i['time']))
            re_index += 1
        else:
            noise_message = ['{}'.format(i['message']) for i in noise_list[noise_id]['noise_message']]
            new_message_list.extend(noise_message)
            noise_id += 1
    
    noisy_traj['message_list'] = new_message_list
    noisy_traj['QA'] = traj['QA']
    
    noisy_traj['QA']['target_step_id'] = [reverse_relocate_dict[step_id[1]] * 3 + step_id[0] for step_id in noisy_traj['QA']['target_step_id']]

    return noisy_traj

def MakeNoiseMessageHighLevel(RawDataPath, Outdir,length, sample_num=100):
    NoiseDataPath = 'NoiseData/ThirdNoise.json'
    NoisePool = load_json(NoiseDataPath)

    output_data = {}

    RawData = load_json(RawDataPath)
    for QAtype, QAtype_data in RawData.items():  
        output_data[QAtype] = {}
        
        for scenario, scenario_data in QAtype_data.items():  
            output_data[QAtype][scenario] = []
            print(QAtype, scenario)
            scenario_traj_all_pre = []
            for traj in scenario_data:
                noisy_traj = infuse_single_trajectory_message_highlevel(traj, NoisePool, length)
                scenario_traj_all_pre.append(noisy_traj)
        
            scenario_traj_all = random.sample(scenario_traj_all_pre, sample_num)
            output_data[QAtype][scenario] = scenario_traj_all
            print(QAtype, scenario, 'Finish!')
    
    print(len(output_data))
    RawDataPathSplit = RawDataPath.split('/') 
    output_path = Outdir + '/' + RawDataPathSplit[1].replace('.json', '') + '_multiple_{}.json'.format(length)
    save_json(output_data, output_path) 

def MakeNoiseMessage(RawDataPath, Outdir,length=10, sample_num=100):
    NoiseDataPath = 'NoiseData/ThirdNoise.json'
    NoisePool = load_json(NoiseDataPath)

    output_data = {}

    RawData = load_json(RawDataPath)
    for QAtype, QAtype_data in RawData.items():  # 获得simple
        output_data[QAtype] = {}
        for scenario, scenario_data in QAtype_data.items():  # 获得role
            output_data[QAtype][scenario] = []
            print(QAtype, scenario)
            scenario_traj_all = []
            for traj in scenario_data:
                noisy_traj = infuse_single_trajectory_message(traj, NoisePool, length)
                # output_data[QAtype][scenario].append(noisy_traj)
                scenario_traj_all.append(noisy_traj)
            
            scenario_traj_all = random.sample(scenario_traj_all, sample_num)
            output_data[QAtype][scenario] = scenario_traj_all
            print(QAtype, scenario, 'Finish!')
    
    RawDataPathSplit = RawDataPath.split('/') 
    output_path = Outdir + '/' + RawDataPathSplit[1].replace('.json', '') + '_multiple_{}.json'.format(length)
    save_json(output_data, output_path)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Make noise for the dataset")
    
    # parser.add_argument("--raw_data_path", type=str, required=True,
    #                     help="原始数据路径")
    # parser.add_argument("--length", type=int, default=10,
    #                     help="噪声长度")
    # parser.add_argument("--sample_num", type=int, default=100,
    #                     help="每个场景采样数量")
    # parser.add_argument("--outdir", type=str, default='data2test',
    #                     help="输出目录")
    
    # args = parser.parse_args()
    
    # MakeNoiseMessage(args.raw_data_path, args.length, args.sample_num, args.outdir)  ## 混合第3人称的lowlevel的message

    # MakeNoiseMessageHighLevel('data/ThirdAgentDataHighLevel.json', 'data2test',length=10, sample_num=100)  ## 混合第3人称的highlevel的message
    # MakeNoiseMessage('data/ThirdAgentDataLowLevel.json', 'data2test', length=10, sample_num=100)  ## 混合第3人称的lowlevel的message
    # MakeNoiseSession('data/FirstAgentDataLowLevel.json', 'data2test', length=10, sample_num=100)  ## 混合第1人称的lowlevel的session
    # MakeNoiseSession('data/FirstAgentDataHighLevel.json', 'data2test', length=10, sample_num=100)  ## 混合第1人称的highlevel的session

    # 1. 生成参与式-事实性数据集 (360条)
    MakeNoiseSession('data/FirstAgentDataLowLevel.json', 'data2test', length=10, sample_num=23)

    # 2. 生成参与式-反思性数据集 (120条)
    MakeNoiseSession('data/FirstAgentDataHighLevel.json', 'data2test', length=10, sample_num=30)

    # 3. 生成观察式-事实性数据集 (280条)
    MakeNoiseMessage('data/ThirdAgentDataLowLevel.json', 'data2test', length=10, sample_num=12)

    # 4. 生成观察式-反思性数据集 (60条)
    MakeNoiseMessageHighLevel('data/ThirdAgentDataHighLevel.json', 'data2test', length=10, sample_num=15)
