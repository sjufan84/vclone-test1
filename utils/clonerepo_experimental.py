import os
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
from pathlib import Path
import requests

def run_script():
    def run_cmd(cmd):
        process = subprocess.run(cmd, shell=True, check=True, text=True)
        return process.stdout

    # Change the current directory to /content/
    os.chdir('/content/')
    print("Changing dir to /content/")

    # Your function to edit the file
    def edit_file(file_path):
        temp_file_path = "/tmp/temp_file.py"
        changes_made = False
        with open(file_path, "r") as file, open(temp_file_path, "w") as temp_file:
            previous_line = ""
            second_previous_line = ""
            for line in file:
                new_line = line.replace("value=160", "value=128")
                if new_line != line:
                    print("Replaced 'value=160' with 'value=128'")
                    changes_made = True
                line = new_line

                new_line = line.replace("crepe hop length: 160", "crepe hop length: 128")
                if new_line != line:
                    print("Replaced 'crepe hop length: 160' with 'crepe hop length: 128'")
                    changes_made = True
                line = new_line

                new_line = line.replace("value=0.88", "value=0.75")
                if new_line != line:
                    print("Replaced 'value=0.88' with 'value=0.75'")
                    changes_made = True
                line = new_line

                if "label=i18n(\"输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络\")" in previous_line and "value=1," in line:
                    new_line = line.replace("value=1,", "value=0.25,")
                    if new_line != line:
                        print("Replaced 'value=1,' with 'value=0.25,' based on the condition")
                        changes_made = True
                    line = new_line

                if "label=i18n(\"总训练轮数total_epoch\")" in previous_line and "value=20," in line:
                    new_line = line.replace("value=20,", "value=500,")
                    if new_line != line:
                        print("Replaced 'value=20,' with 'value=500,' based on the condition for DEFAULT EPOCH")
                        changes_made = True
                    line = new_line

                if 'choices=["pm", "harvest", "dio", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny"], # Fork Feature. Add Crepe-Tiny' in previous_line:
                    if 'value="pm",' in line:
                        new_line = line.replace('value="pm",', 'value="mangio-crepe",')
                        if new_line != line:
                            print("Replaced 'value=\"pm\",' with 'value=\"mangio-crepe\",' based on the condition")
                            changes_made = True
                        line = new_line

                new_line = line.replace('label=i18n("输入训练文件夹路径"), value="E:\\\\语音音频+标注\\\\米津玄师\\\\src"', 'label=i18n("输入训练文件夹路径"), value="/content/dataset/"')
                if new_line != line:
                    print("Replaced 'label=i18n(\"输入训练文件夹路径\"), value=\"E:\\\\语音音频+标注\\\\米津玄师\\\\src\"' with 'label=i18n(\"输入训练文件夹路径\"), value=\"/content/dataset/\"'")
                    changes_made = True
                line = new_line

                if 'label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),' in second_previous_line:
                    if 'value=i18n("否"),' in line:
                        new_line = line.replace('value=i18n("否"),', 'value=i18n("是"),')
                        if new_line != line:
                            print("Replaced 'value=i18n(\"否\"),' with 'value=i18n(\"是\"),' based on the condition for SAVE ONLY LATEST")
                            changes_made = True
                        line = new_line

                if 'label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),' in second_previous_line:
                    if 'value=i18n("否"),' in line:
                        new_line = line.replace('value=i18n("否"),', 'value=i18n("是"),')
                        if new_line != line:
                            print("Replaced 'value=i18n(\"否\"),' with 'value=i18n(\"是\"),' based on the condition for SAVE SMALL WEIGHTS")
                            changes_made = True
                        line = new_line

                temp_file.write(line)
                second_previous_line = previous_line
                previous_line = line

        # After finished, we replace the original file with the temp one
        import shutil
        shutil.move(temp_file_path, file_path)

        if changes_made:
            print("Changes made and file saved successfully.")
        else:
            print("No changes were needed.")

    # Define the repo path
    repo_path = '/content/Applio-RVC-Fork'

    def copy_all_files_in_directory(src_dir, dest_dir):
        # Iterate over all files in source directory
        for item in Path(src_dir).glob('*'):
            if item.is_file():
                # Copy each file to destination directory
                shutil.copy(item, dest_dir)
            else:
                # If it's a directory, make a new directory in the destination and copy the files recursively
                new_dest = Path(dest_dir) / item.name
                new_dest.mkdir(exist_ok=True)
                copy_all_files_in_directory(str(item), str(new_dest))

    def clone_and_copy_repo(repo_path):
        # New repository link
        new_repo_link = "https://github.com/IAHispano/Applio-RVC-Fork/"
        # Temporary path to clone the repository
        temp_repo_path = "/content/temp_Applio-RVC-Fork"
        # New folder name
        new_folder_name = "Applio-RVC-Fork"

        # Clone the latest code from the new repository to a temporary location
        run_cmd(f"git clone {new_repo_link} {temp_repo_path}")
        os.chdir(temp_repo_path)

        run_cmd(f"git checkout 3fa4dad3d8961e5ca2522e9e12c0b4ddb71ad402")
        run_cmd(f"git checkout f9e606c279cb49420597519b0a83b92be81e42e4")
        run_cmd(f"git checkout 9e305588844c5442d58add1061b29beeca89d679")
        run_cmd(f"git checkout bf92dc1eb54b4f28d6396a4d1820a25896cc9af8")
        run_cmd(f"git checkout c3810e197d3cb98039973b2f723edf967ecd9e61")
        run_cmd(f"git checkout a33159efd134c2413b0afe26a76b7dc87926d2de")
        run_cmd(f"git checkout 24e251fb62c662e39ac5cf9253cc65deb9be94ec")
        run_cmd(f"git checkout ad5667d3017e93232dba85969cddac1322ba2902")
        run_cmd(f"git checkout ce9715392cf52dd5a0e18e00d1b5e408f08dbf27")
        run_cmd(f"git checkout 7c7da3f2ac68f3bd8f3ad5ca5c700f18ab9f90eb")
        run_cmd(f"git checkout 4ac395eab101955e8960b50d772c26f592161764")
        run_cmd(f"git checkout b15b358702294c7375761584e5276c811ffab5e8")
        run_cmd(f"git checkout 1501793dc490982db9aca84a50647764caa66e51")
        run_cmd(f"git checkout 21f7faf57219c75e6ba837062350391a803e9ae2")
        run_cmd(f"git checkout b5eb689fbc409b49f065a431817f822f554cebe7")
        run_cmd(f"git checkout 7e02fae1ebf24cb151bf6cbe787d06734aa65862")
        run_cmd(f"git checkout 6aea5ea18ed0b9a1e03fa5d268d6bc3c616672a9")
        run_cmd(f"git checkout f0f9b25717e59116473fb42bd7f9252cfc32b398")
        run_cmd(f"git checkout b394de424088a81fc081224bc27338a8651ad3b2")
        run_cmd(f"git checkout f1999406a88b80c965d2082340f5ea2bfa9ab67a")
        run_cmd(f"git checkout d98a0fa8dc715308dfc73eac5c553b69c6ee072b")
        run_cmd(f"git checkout d73267a415fb0eba98477afa43ef71ffd82a7157")
        run_cmd(f"git checkout 1a03d01356ae79179e1fb8d8915dc9cc79925742")
        run_cmd(f"git checkout 81497bb3115e92c754300c9b3992df428886a3e9")
        run_cmd(f"git checkout c5af1f8edcf79cb70f065c0110e279e78e48caf9")
        run_cmd(f"git checkout cdb3c90109387fa4dfa92f53c3864c71170ffc77")

        # Edit the file here, before copying
        #edit_file(f"{temp_repo_path}/infer-web.py")

        # Copy all files from the cloned repository to the existing path
        copy_all_files_in_directory(temp_repo_path, repo_path)
        print(f"Copying all {new_folder_name} files from GitHub.")

        # Change working directory back to /content/
        os.chdir('/content/')
        print("Changed path back to /content/")
        
        # Remove the temporary cloned repository
        shutil.rmtree(temp_repo_path)

    # Call the function
    clone_and_copy_repo(repo_path)

    # Download the credentials file for RVC archive sheet
    os.makedirs('/content/Applio-RVC-Fork/stats/', exist_ok=True)
    run_cmd("wget -q https://cdn.discordapp.com/attachments/945486970883285045/1114717554481569802/peppy-generator-388800-07722f17a188.json -O /content/Applio-RVC-Fork/stats/peppy-generator-388800-07722f17a188.json")

    # Forcefully delete any existing torchcrepe dependencies downloaded from an earlier run just in case
    shutil.rmtree('/content/Applio-RVC-Fork/torchcrepe', ignore_errors=True)
    shutil.rmtree('/content/torchcrepe', ignore_errors=True)

    # Download the torchcrepe folder from the maxrmorrison/torchcrepe repository
    run_cmd("git clone https://github.com/maxrmorrison/torchcrepe.git")
    shutil.move('/content/torchcrepe/torchcrepe', '/content/Applio-RVC-Fork/')
    shutil.rmtree('/content/torchcrepe', ignore_errors=True)  # Delete the torchcrepe repository folder

    # Change the current directory to /content/Applio-RVC-Fork
    os.chdir('/content/Applio-RVC-Fork')
    os.makedirs('pretrained', exist_ok=True)
    os.makedirs('uvr5_weights', exist_ok=True)

def download_file(url, filepath):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

def download_pretrained_models():
    pretrained_models = {
        "pretrained": [
            "D40k.pth",
            "G40k.pth",
            "f0D40k.pth",
            "f0G40k.pth"
        ],
        "pretrained_v2": [
            "D40k.pth",
            "G40k.pth",
            "f0D40k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
            "f0D48k.pth"
        ],
        "uvr5_weights": [
            "HP2-人声vocals+非人声instrumentals.pth",
            "HP5-主旋律人声vocals+其他instrumentals.pth",
            "VR-DeEchoNormal.pth",
            "VR-DeEchoDeReverb.pth",
            "VR-DeEchoAggressive.pth",
            "HP5_only_main_vocal.pth",
            "HP3_all_vocals.pth",
            "HP2_all_vocals.pth"
        ]
    }
    part2 = "I"
    base_url = "https://huggingface.co/lj1995/VoiceConversionWebU" + part2 + "/resolve/main/"
    base_path = "/content/Applio-RVC-Fork/"
    base_pathm = base_path

    # Calculate total number of files to download
    total_files = sum(len(files) for files in pretrained_models.values()) + 1  # +1 for hubert_base.pt

    with tqdm(total=total_files, desc="Downloading files") as pbar:
        for folder, models in pretrained_models.items():
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            for model in models:
                url = base_url + folder + "/" + model
                filepath = os.path.join(folder_path, model)
                download_file(url, filepath)
                pbar.update()

        # Download hubert_base.pt to the base path
        hubert_url = base_url + "hubert_base.pt"
        hubert_filepath = os.path.join(base_pathm, "hubert_base.pt")
        download_file(hubert_url, hubert_filepath)
        pbar.update()
def clone_repository(run_download):
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(run_script)
        if run_download:
            executor.submit(download_pretrained_models)
