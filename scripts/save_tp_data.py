#coding=utf-8
import os
import numpy as np

# dn 1463, tpn 2042
if __name__ == "__main__":
    xbxs = ['PC3', 'VCAP', 'A375', 'A549', 'HA1E', 'HCC515', 'HEPG2']
    # xbxs = ['A375']
    dir_path = "./drug_data/trt_sh/"

    for xbx in xbxs:
        # 这个是每个xbx的real negative的实验id
        rn = []
        file_read_name = './real_negative_922/rn_' + xbx + '.txt'
        file_read = open(file_read_name, 'r')
        lines = file_read.readlines()
        file_read.close()
        print xbx, 'rn len:', len(rn)
        # print "rn:", rn[: 10]

        exp_ids = []
        qiguaide_ids = []
        for items in lines:
            item = items.replace('\n','').split('\t')
            exp_ids.append(item[0])
            qiguaide_ids.append(item[1])

        # 遍历不同的serise文件
        count = 0
        lines = []
        qiguaide_lines = []
        for root,dirs,files in os.walk(dir_path+xbx): # 遍历某个xbx下的所有serise文件
            for file in files:
                print '    processing:', os.path.join(root,file)
                with open( os.path.join(root,file) ) as file_p:
                    file_lines = file_p.readlines()
                file_lines = [file_line[: -2].split('\t') for file_line in file_lines] # list类型!   -2:文件结尾是\r\n
                for shiyan_id in xrange(len(file_lines[0])): # 遍历每个shiyan id
                    # print len(file_lines[0][shiyan_id])
                    # raw_input("press any key...")
                    if file_lines[0][shiyan_id] in exp_ids: # 如果该实验id属于rn里面,该列则保存到tp_data中
                        count += 1
                        line = []
                        for i in file_lines[1:]: # 不要第一行的实验id
                            line.append(i[shiyan_id+1]) # 实验id要加1,因为第一行的列数是少了1的
                        lines.append(line)
                        qiguaide_lines.append(qiguaide_ids[exp_ids.index(file_lines[0][shiyan_id])]+'\n')
        lines = np.array(lines)
        print '    lines.shape:', lines.shape
        np.save('real_negative_922/tp_data_'+xbx+'.npy', lines) # 应该是tp_data_
        file_write = open('real_negative_922/tp_data_'+xbx+'_giguaidelie.txt', 'w')
        file_write.writelines(qiguaide_lines) # 如果读取的时候以"\t"分割要记得去掉最后一个, 如果写"\n"要记得每一行都要去掉"\n"
        file_write.close()

        print '   ', xbx, 'done!', count, '个rn~\n'
