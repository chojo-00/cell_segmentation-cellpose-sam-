import numpy as np

file_path = 'result/im_20230921114633_037_568088337/im_20230921114633_037_568088337_seg.npy'  # 파일 경로를 본인 환경에 맞게 수정
data = np.load(file_path, allow_pickle=True)

print(type(data))
print(data.shape)

# 내용 미리 보기
print(data)

import numpy as np

import numpy as np

data = np.load('result/im_20230921114633_037_568088337/im_20230921114633_037_568088337_seg.npy', allow_pickle=True)

# 내부 dict 객체 꺼내기
data = data.item()  # 또는 data[()] 도 가능

# 이제 정상적으로 접근 가능
mask = data['masks']
print("세포 개수:", len(np.unique(mask)) - 1)

cell_ids = np.unique(mask)
cell_ids = cell_ids[cell_ids != 0]  # 배경 제외

areas = {cid: np.sum(mask == cid) for cid in cell_ids}