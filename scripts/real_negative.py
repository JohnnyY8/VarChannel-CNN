#coding=utf-8

# dn 1463, tpn 2042
if __name__ == "__main__":
    f = open('negative_target.txt', 'r')
    negative_target = []
    lines = f.readlines()
    for line in lines:
        negative_target.append((line.split('\r\n'))[0])
    f.close()
    print 'negative_target done, len:', len(negative_target)

    xbxs = ['PC3', 'VCAP', 'A375', 'A549', 'HA1E', 'HCC515', 'HEPG2']
    # 新建相应的文件
    for xbx in xbxs:
        file_write_name = 'real_negative_922/rn_' + xbx + '_temp.txt'
        file_write = open(file_write_name, 'w')
        file_write.close()

    f = open('trt_sh.info', 'r')
    number_lines = len(f.readlines())
    f.close()
    f = open('trt_sh.info', 'r')
    line = f.readline() # 不要第一行
    line = f.readline()
    i = 2

    while line:
        line = line.split('\t')
        if line[3] in xbxs and line[2] in negative_target: # 如果他是我们需要的细胞系并且一定是负样本,就存下来
            file_write_name = 'real_negative_922/rn_' + line[3] + '_temp.txt'
            file_write = open(file_write_name, 'a')
            file_write.write(line[0]+'\t'+line[2]+'\n') # 如果读取的时候以"\t"分割要记得去掉最后一个, 如果写"\n"要记得每一行都要去掉"\n"
            file_write.close()
        line = f.readline()
        if i%10000 == 0:
            print i, '/', number_lines
        i += 1
    f.close()
    print 'done'

    for xbx in xbxs:
        rn = []
        # file_read_name = 'real_negative/rn_' + xbx + '_temp.txt'
        file_read_name = 'real_negative_922/rn_' + xbx + '_temp.txt'
        file_read = open(file_read_name, 'r')
        line = file_read.readlines()
        print xbx, ':', len(line)

        file_write_name = 'real_negative_922/rn_' + xbx + '.txt'
        file_write = open(file_write_name, 'w')
        file_write.writelines(list(set(line)))
        file_write.close()
        print '去重后的', xbx, ':', len(list(set(line)))

    # PC3 : 41300
    # 去重后的 PC3 : 41299
    # VCAP : 43716
    # 去重后的 VCAP : 43715
    # A375 : 27921
    # 去重后的 A375 : 27920
    # A549 : 26845
    # 去重后的 A549 : 26844
    # HA1E : 27223
    # 去重后的 HA1E : 27222
    # HCC515 : 24099
    # 去重后的 HCC515 : 24098
    # HEPG2 : 23876
    # 去重后的 HEPG2 : 23875
