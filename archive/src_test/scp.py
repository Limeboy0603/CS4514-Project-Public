import os

# list all directories in ./frames
dirs = os.listdir('./dataset/tvb-hksl-news/keypoints')
# remove everything before '2020-04-04'
dirs = [d for d in dirs if d > '2020-05-17']
dirs = sorted(dirs)
print(dirs)

password = '' # your password here
target_dir = './dataset/tvb-hksl-news/frames'

for d in dirs:
    src_dir = '/public/dickmwong3/dataset/tvb-hksl-news/frames/' + d
    print(src_dir)
    os.system(f'sshpass -p {password} scp -r dickmwong3@gateway.cs.cityu.edu.hk:{src_dir} {target_dir}')
    os.system(f'sshpass -p {password} ssh dickmwong3@gateway.cs.cityu.edu.hk "rm -rf {src_dir}"')
