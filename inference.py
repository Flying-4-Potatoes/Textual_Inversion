from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
#model_id = "textual_inversion_fairy_tale"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16,revision="fp16").to("cuda")

#prompt = "A photo of Illustration, dungeon, knife, blonde, 20 years old, woman "
#prompt = "Illustration, dungeon, knife, blonde, 20 years old, woman <illustrate-fairy tale>."

def generate_img(caption,num):
    image = pipe(prompt=caption).images[0]
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(caption[:5] + '_' + str(num) + ".svg", format="svg", bbox_inches="tight")
    #image = pipe(prompt=caption).images[0]
    #image.save(caption + '_' + model_name + ".png")
    #plt.savefig(image)
    #image.save(caption[:5] + '_' + str(num) + ".png")
    #plt.imshow(image)
    #plt.show()
    
#special_token = 'cinematic, dramatic, beautiful, composition, hyper realistic, epic scale, sense of awe, hypermaximalist, insane level of details, trending on artstation HQ, art by artgerm and greg rutkowski'



def main():
    
  for i in range(1,11):
      generate_img('a photo of city landscape with a steampunk robot and a machine. high detailed, insane quality, 4k',i)

    


if __name__ == '__main__':
    main()

#prompt = "Hello, I want to write an illustration on the cover of the webtoon, please draw the picture. The picture is about the secrets of Dungeon and Treasure, and the illustration I need is a blonde 20-year-old female knight with a knife passing through the drawer."




#image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]

#plt.imshow(image)
#plt.show()


#art by artgerm and greg rutkowski

#"A cover art of A solemn fair princess gazing out castle window, fantasy character portrait, fall woodland, ultra realistic, intricate, elegant, highly detailed, digital painting, artstaion, smooth, sharp, focus, illustration, cinematic, dramatic, beautiful, composition, hyper realistic, epic scale, sense of awe, hypermaximalist, insane level of details, trending on artstation HQ"

#"Beautiful illustration of a big castle in a serene landscape with a knight standing nearby, by albert bierstadt, green grass, highly detailed, crystal lighting, mystical, forest, hyperrealistic, 4 k, unreal engine, magical, by joe fenton, by greg rutkowski, by greg tocchini, by kaws, by kate beaton, by kaethe butcher"

#"The Witch fed Snow White a poisoned apple."