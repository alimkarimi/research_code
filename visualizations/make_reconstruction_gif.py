import glob
from PIL import Image

def make_gif(frame_folder):
    #print(frame_folder)
    #print(glob.glob(f"{frame_folder}/*.jpg"))
    glob_obj = glob.glob(f"{frame_folder}/*.jpg")
    glob_obj.sort()
    frames = [Image.open(image) for image in glob_obj]
    
    #print(frames)
    print(frames[0])
    print('sorted:', glob_obj)
    frame_one = frames[0]
    frame_one.save("autoencoder_reconstruction.gif", format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)
    
if __name__ == "__main__":
    make_gif('/Users/alim/Documents/prototyping/research_lab/research_code/visualizations/ae_reconstructions')