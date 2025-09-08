import torch.distributed as dist

def dprint(*args, **kwargs):
    rank=dist.get_rank()
    if rank==0:
        print(*args, **kwargs)





def wprint(*args, **kwargs):
    rank = dist.get_rank()
    if rank == 0:
        # 打印到控制台
        print(*args, **kwargs)
        # 要写入的文件名
        filename = 'console.txt'
        # 打开文件（追加模式）
        with open(filename, 'a') as f:
            # 写入内容到文件
            # 先将 args 转换为字符串
            output_str = ' '.join(map(str, args))
            # 写入到文件
            f.write(output_str + '\n')
            # 如果 kwargs 中有 end 参数，使用它，否则使用默认的换行
            if 'end' in kwargs:
                f.write(kwargs['end'])
            else:
                f.write('\n')



# # 在代码中使用
# debug_print("这条消息只会在随机选择的调试节点显示")
