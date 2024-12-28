
# original 34 markers list
marker_name_list = [
    "RFHD", "LFHD", "RBHD", "LBHD", 
    "C7", "T10", "CLAV", "STRN",
    "RSHO", "RELB", "RWRA", "RWRB", "RFIN", 
    "LSHO", "LELB", "LWRA", "LWRB", "LFIN",
    "RASI", "LASI", "RPSI", "LPSI", 
    "RKNE", "RHEE", "RTOE", "RANK", 
    "LKNE", "LHEE", "LTOE", "LANK",
    "VIOLINE", "VIOLINB", "BOWE", "BOWB"
]

# combine markers to joint list
new_joint_list = [
    ['Head', 0, 12], ['Neck', 12, 18], ['Root', 54, 66],
    ['Rshoulder', 24, 27], ['Relbow', 27, 30], ['Rwrist', 30, 36], ['Rfinger', 36, 39],
    ['Lshoulder', 39, 42], ['Lelbow', 42, 45], ['Lwrist', 45, 51], ['Lfinger', 51, 54],
    ['Rknee', 66, 69], ['Rhee', 69, 72], ['Rank', 75, 78], ['Rtoe', 72, 75], 
    ['Lknee', 78, 81], ['Lhee', 81, 84], ['Lank', 87, 90], ['Ltoe', 84, 87],
    ['Vtop', 90, 93], ['Vbom', 93, 96], ['Btop', 96, 99], ['Bbom', 99, 102],
    ['Torso', 18, 24]
]

# output 21 joint data
joint_name_list =  [
    'Head','Neck','Root', 
    'Rshoulder', 'Relbow','Rwrist','Rfinger',
    'Lshoulder','Lelbow', 'Lwrist', 'Lfinger',
    'Rknee','Rank','Rtoe', 
    'Lknee','Lank','Ltoe',
    'Vtop','Vbom','Btop', 'Bbom',
    'Torso'
]

# the reference joints used for normalization
norm_ref_list = [['Rtoe', 72, 75], ['Ltoe', 84, 87]]
