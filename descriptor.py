import os
import argparse
import random
from tqdm import tqdm
import inflect
import json
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer,  CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from accelerate import Accelerator

random.seed(0)


class Descriptor:
    """
    This class is designed to generate and train a sentence that describes the object present in an image.
    """

    def __init__(self, image_path, workspace="example", device="cuda", num_words = 15, verbose = True, model_key = "stabilityai/stable-diffusion-2-1-base"):
        """
        Initializes the ImageDescriptionGenerator class by loading all necessary models and variables.

        Args:
            image_path (str): The path to the image that contains the object to be described.
            workspace (str): The path to the directory where all files will be created.
            device (str): The device (CPU or GPU) to be used for computations.
            num_words (int): The maximum number of words to be used in the sentence describing the object.
            verbose (bool): Set to True if you want to see detailed information during the execution.
            model_key (str): The key indicating which Stable Diffusion model should be loaded.
        """

        # Load nouns dictionary.
        with open('nounlist.txt') as f:
            lines = f.readlines()
        self.nouns = [word[:-1] for word in lines]

        if verbose:
            print("LOADED: A list with common english nouns of length: " + str(len(self.nouns)))

        # Load adjectives dictionary
        with open('adjectives.txt') as f:
            lines = f.readlines()
        self.adjectives = [word[1:-2] for word in lines]

        if verbose:
            print("LOADED: A list with common english adjectives of length: " + str(len(self.adjectives)))

        # Dictionary that contains both nouns and adjectives
        self.nouns_adjectives = self.nouns + self.adjectives

        # Dictionary with quantificators used to describe the noun quantity.
        self.quantificators = ["a", "an", "some", "a few", "a little", "a lot of", "much", "many"]
        if verbose:
            print("LOADED: A list with quantificators of lentgh: " + str(len(self.quantificators)))

        # Create all the directories
        if not os.path.exists(workspace):
            os.mkdir(workspace)
            os.makedirs(workspace + "/descriptor/tokens")
            os.makedirs(workspace + "/descriptor/samples")
        if not os.path.exists(workspace + "/descriptor/tokens"):
            os.makedirs(workspace + "/descriptor/tokens")
        if not os.path.exists(workspace + "/descriptor/samples"):
            os.makedirs(workspace + "/descriptor/samples")

        # Load all the transformations and compose them. This will be used for data augmentation purposes.
        transform1 = transforms.RandomHorizontalFlip(p=0.5)
        transform2 = transforms.RandomApply([transforms.RandomRotation(degrees=10, fill=1.0)], p=0.9)
        transform3 = transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.25)
        transform4 = transforms.RandomApply([transforms.RandomCrop(512 * 0.8, fill=1.0)], p=0.5)
        transform5 = transforms.RandomGrayscale(p=0.10)
        transform6 = transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)])
        transform7 = transforms.Resize(512)
        self.trans = transforms.Compose([transform1, transform2, transform3, transform4, transform5, transform6, transform7])

        # Load class variables.
        self.verbose = verbose
        self.device = device
        self.num_words = num_words
        self.workspace = workspace
        self.model_key = model_key

        # Load the image and resize it to be able to fit inside the models.
        img = (torchvision.io.read_image(image_path).float() / 255.).unsqueeze(0).to(self.device)
        img = F.interpolate(img, (512, 512), mode='bilinear', align_corners=False)
        self.img = img

        # Variable part of the sentence building that will be used later.
        self.noun = ""
        self.quantifier = ""
        self.words_list = ""
        self.placeholder_token_id = -1


    def get_base_sentence(self,  noun, quantifier, model_path = "openai/clip-vit-base-patch32", processor_path = "openai/clip-vit-base-patch32"):
        """
        Generates a base sentence that can be used to describe the image.

        Args:
            name (str): The base name describing the object. Optional, if not provided will be automatically calculated.
            quantifier (str): The base quantifier describing the object. Optional, if not provided will be automatically calculated.
            image_path (str): The system image path.
            model_path (str): The path to load the CLIP model.
            processor_path (str): The path to load the CLIP processor.

        Returns:
            None. The generated sentence is stored internally for later use.
        """

        # Load CLIP. Model used for inner product and processor to prepare previously the data.
        model = CLIPModel.from_pretrained(model_path).to(self.device)
        processor = CLIPProcessor.from_pretrained(processor_path)

        # Get the embedding of the image.
        img_embedding= processor(images=self.img[0], return_tensors="pt", padding=True)

        # THis value is used as maximum obtained similarity score. Used to stop the execution if it doesn't improve.
        value_max = 0

        # This block of code adds a noun if it hasn't been provided.
        if noun is None:
            if self.verbose: print("[INFO] Searching NOUN...", end="\r")
            quantifier, noun, value_max = self.get_noun(model, processor, img_embedding)
            if self.verbose: print("[INFO] Found NOUN, sentence is: ", Descriptor.get_sentence([], noun, quantifier), "  ||  Value: ", value_max)

        # This block of code adds words until the limit number is reached.
        words_list = []
        added_word = True  # Used as a condition inside the while block. Exits if we haven't added new word.
        while (len(words_list) < self.num_words) and added_word:
            added_word = False

            # This block of code adds a descriptor (word used to describe the object in the image) if adding it imrpoves the similarity score.
            if self.verbose: print("[INFO] Searching DESCRIPTOR...", end="\r")
            word, value = self.add_word(model, processor, words_list, noun, quantifier, img_embedding)
            if value > value_max:
                value_max = value
                words_list.append(word)
                if self.verbose: print("[INFO] Found DESCRIPTOR, sentence is: ", Descriptor.get_sentence(words_list, noun, quantifier),
                                  "  ||  Value: ", value_max)
                added_word = True
            # If adding a word doesn't imrpove the similarity score the execution is over.
            elif self.verbose:
                print("[INFO] No more words are needed, sentence is: ", Descriptor.get_sentence(words_list, noun, quantifier),
                      "  ||  Value: ", value_max)

        # This block of code adjusts the words. It chooses a random added word and looks if changing it improves the similarity score.
        if self.verbose: print("[INFO] Checking if changes are needed...", end="\r")
        value_max = self.adjust_words(model, processor, words_list, noun, quantifier, img_embedding, value_max)

        if self.verbose: print("[INFO] Changes have been checked, sentence is: ", Descriptor.get_sentence(words_list, noun, quantifier),
                          "  ||  Value: ", value_max)

        # Final obtained words that describe our image.
        self.quantifier = quantifier
        self.noun = noun
        self.words_list = words_list
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()

        if self.verbose: print("[INFO] Done searching base sentence")


    def get_ids(self, sentence):
        """
        This method returns the ids of the tokens that will be updated.

        Args:
            sentence: which tokens ids will be extracted

        Returns: List with the ids
        """

        # First get the token's corresponding numbers
        text_input = self.tokenizer(sentence, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True,
                               return_tensors='pt')

        # Load in a list the tokens. Stops when a 0, that corresponds to no token, is encountered
        tokens_num = text_input['input_ids'][0]
        for i in range(len(tokens_num)):
            if tokens_num[i] == 0:
                break;

        # Removes the first and last token as their correspond to indicate begin and end of sentence
        return tokens_num[1:i - 1]

    def save(self, name):
        """
        Stores the token undr workspace/tokens/name

        Args:
            name: The name of the file that will be saved

        Returns: None
        """
        learned_embeds = self.text_encoder.get_input_embeddings().weight[self.placeholder_token_id].data
        torch.save(learned_embeds, name)

    def fit_description(self, chk_every=1000, print_every=2500, iterations=10000, samples=5, load=-1):
        """
        Trains the tokens and saves them.

        Args:
            chk_every (int): Number of iterations between saves.
            print_every (int): Number of iterations between image prints (saved in workspace/descriptor/samples)
            iterations (int): Total number of training iterations.
            samples (int): Number of samples to generate when printing images.

        Returns:
            None. Token embeddings are stored.
        """

        # Load all the models
        self.vae = AutoencoderKL.from_pretrained(self.model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.model_key, subfolder="unet").to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        self.alphas_cum = self.scheduler.alphas_cumprod.to(self.device)
        self.alphas = self.scheduler.alphas.to(self.device)
        self.accelerator = Accelerator()

        # Set models as not trainable except the CLIPs word embedding part
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Load from checkpoint if needed.
        self.load_from_ckp(load)
        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[self.placeholder_token_id] = False

        # Get a base descriptor sentence. Extract the second only with the noun
        self.sentence = Descriptor.get_sentence(self.words_list, self.noun, self.quantifier)
        self.sentence2 = "An image of " + self.quantifier + " " + self.noun

        # Optimizer and loss used during training
        optimizer = torch.optim.Adam(self.text_encoder.get_input_embeddings().parameters(), lr=3e-4)
        loss_fn = torch.nn.MSELoss()

        # Load the tokens
        self.text_encoder.train()
        text_input = self.tokenizer(self.sentence, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        text_input2 = self.tokenizer(self.sentence2, padding='max_length', max_length=self.tokenizer.model_max_length,
                                     truncation=True,
                                     return_tensors='pt')

        # Train
        for iteration in tqdm(range(self.iteration_begin + 1, iterations)):
            # Checkpoint. Save the tokens
            if iteration % chk_every == 0:
                self.save("./"+self.workspace+"/descriptor/tokens/" + self.noun + "_" + str(iteration)+".pt")
                dictionary = {
                    "iteration": iteration,
                    "placeholder": self.placeholder_token_id,
                    "noun": self.noun,
                    "quantifier": self.quantifier,
                    "words_list": self.words_list,
                }
                # Serializing json
                json_object = json.dumps(dictionary, indent=4)
                with open("./" + self.workspace + "/descriptor/status.txt", "w") as outfile:
                    outfile.write(json_object)
            # Generate and store images for visualization purposes
            if iteration % print_every == 0:
                self.sample(iteration, samples)

            with torch.no_grad():

                # Sample time
                t = torch.randint(1000, [1], device=self.device)

                # Sample noise
                noise = torch.randn([1, 4, 64, 64]).to(self.device)

                # Sample image with noise
                latents = self.encode_imgs(self.trans(self.img))  # Remove if not with trans
                xt = self.scheduler.add_noise(latents, noise, t)
                del latents

                text_embeddings = self.text_encoder(text_input2.input_ids.to(self.device))[0]
                noise_pred_base = self.unet(xt, t, encoder_hidden_states=text_embeddings).sample

            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            noise_pred = self.unet(xt, t, encoder_hidden_states=text_embeddings).sample

            # The noise prediction should be the same or very similar
            loss = loss_fn(noise, noise_pred) + loss_fn(noise_pred, noise_pred_base)

            # Update the model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Set the embeddings for not trainable words as they were before the training step
            with torch.no_grad():
                self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = self.orig_embeds_params[index_no_updates]

        if self.verbose:
            print("Pretraining done!")


    def sample(self, iteration, number = 5):
        """
        Generates samples (images) by utilizing trained tokens with the obtained sentence and stores them.

        Args:
            iteration (int): An integer indicating the training iteration. Used to name the generated images.
            number (int, optional): The number of samples to be generated. Default value is 5.

        Returns:
            None. Stores image in file system under the directory ./workspace/descriptor/samples
        """

        # First load the text embedding calculated from the model that is trained.
        text_input = self.tokenizer("", padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True,
                               return_tensors='pt')
        with torch.no_grad():
            no_info = self.text_encoder(text_input.input_ids.to(self.device))[0]

        text_input = self.tokenizer(self.sentence, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True,
                               return_tensors='pt')
        with torch.no_grad():
            embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        text_embed = torch.cat([no_info, embed])

        for i in range(number):
            # Sample random noise
            xt = torch.randn([1, 4, 64, 64]).to(self.device)

            with torch.no_grad():
                guidance = 10  # Indicates how much impact does the text have on the generation
                step = 10  # Indicate how much diffusion backward steps are done altogether
                alphas_cum = self.alphas_cum
                for t in range(999, 0, -step):
                    xt = torch.cat([xt] * 2)

                    # First predict the noise using the text embedding
                    with torch.no_grad():
                        noise_pred = self.unet(xt, t, encoder_hidden_states=text_embed).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_text + guidance * (noise_pred_text - noise_pred_uncond)

                    # Generate the previous, less nosied, version of the image
                    if t - step > 0:
                        xt = (alphas_cum[t - step] ** 0.5) * (xt[0] - ((1 - alphas_cum[t]) ** 0.5) * noise_pred) / (
                                    alphas_cum[t] ** 0.5) + ((1 - alphas_cum[t - step]) ** 0.5) * noise_pred
                    else:
                        xt = (xt[0] - ((1 - alphas_cum[t]) ** 0.5) * noise_pred) / (alphas_cum[t] ** 0.5)

                    # If this is the last step store the image
                    if t - step < 1:
                        img = self.decode_latents(xt)
                        save_image(img, "./"+self.workspace+"/descriptor/samples/" + str(iteration) + "_" + str(i) + ".png")

    def load_from_ckp(self, load_numb):
        """
        Loads the information from saved files.

        Args:
            load_numb (int): The token checkpoint to be used for loading. If not provided the last version is loaded.

        Returns:
            None. The model is laoded.
        """

        # If status file doesn't exists, it is created
        if not os.path.exists("./" + self.workspace + "/descriptor/status.txt"):
            self.placeholder_token_id = ((self.get_ids(" ".join(self.words_list))).cpu()).tolist()
            self.orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()
            dictionary = {
                "iteration": 0,
                "placeholder": ((self.get_ids(" ".join(self.words_list))).cpu()).tolist(),
                "noun": self.noun,
                "quantifier": self.quantifier,
                "words_list": self.words_list,
            }

            self.iteration_begin = 0

            # Serializing json
            json_object = json.dumps(dictionary, indent=4)

            with open("./" + self.workspace + "/descriptor/status.txt", "w") as outfile:
                outfile.write(json_object)
            return False

        # If the status file exxists the information from it and the token checkpoints are loaded.
        else:
            data = json.load(open("./" + self.workspace + "/descriptor/status.txt"))
            self.iteration_begin = data["iteration"]
            self.placeholder_token_id  = data["placeholder"]
            self.noun = data["noun"]
            self.quantifier = data["quantifier"]
            self.words_list = data["words_list"]
            self.orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()

            if self.iteration_begin != 0:
                if load_numb != -1:
                    self.iteration_begin = load_numb
                path = "./"+self.workspace+"/descriptor/tokens/" + self.noun + "_" + str(self.iteration_begin) + ".pt"
                with torch.no_grad():
                    embeddings = self.text_encoder.get_input_embeddings()
                    embeddings.weight[self.placeholder_token_id] = torch.load(path)
                    self.text_encoder.set_input_embeddings(embeddings)
            return True

    def get_text_descriptor(self, model, processor, text_list, img_embedding):
        """
        Selects the best matching text from a list of texts based on the image embedding.

        Args:
            model: The model used to calculate the matching values.
            processor: The model used to obtain the embeddings.
            text_list (list): The list of texts to be evaluated.
            img_embedding: The embedding of the image.

        Returns:
            Tuple[int, float]: The position of the best match and the obtained cosine similarity.
        """

        # Lists are employed to store various temporal information since the data cannot be fed all at once due to size limitations.
        maximum = []
        argmaximum = []
        for i in range((len(text_list) // 1000) + 1):
            # Chunk the text.
            tmp = text_list[1000 * i:min(1000 * (i + 1), len(text_list) + 1)]
            with torch.no_grad():
                # Obtain the text embeddings.
                text_inputs = processor(text=tmp, return_tensors="pt", padding=True)
                outputs = model(input_ids=text_inputs['input_ids'].to(self.device),
                                attention_mask=text_inputs['attention_mask'].to(self.device),
                                pixel_values=img_embedding['pixel_values'].to(self.device))
                # Obtain the logits that indicate the similarity score.
                logits_per_image = outputs.logits_per_image
                # Append the maximum value obtained this block.
                maximum.append(logits_per_image.max())
                # Append where it's located.
                argmaximum.append(1000 * i + logits_per_image.argmax())

        return argmaximum[torch.tensor(maximum).argmax()], torch.tensor(maximum).max()

    def get_sentence(words_list, noun=None, quantifier=None, view=None):
        """
        Builds a sentence containing the noun, quantifier, view if provided, and words list.

        Args:
            words_list (list): The list of words to be used.
            noun (str): The noun to be included in the sentence.
            quantifier (str): A quantifier word, such as "a", "an", "many", etc.
            view (str): A word or phrase representing a view, such as "front", "side", "back", etc. Optional.

        Returns:
            str: The generated sentence.
        """

        sentence = "An image of "
        if quantifier: sentence += quantifier + " "
        for word in words_list:
            sentence += word + " "
        sentence = sentence[:-1]  # Remove the last space
        if noun: sentence += " " + noun
        if view: sentence += ", " + view + " view"
        return sentence

    def get_noun(self, model, processor, img_encoding):
        """
        Returns the best quantifier + noun match based on the image embedding.

        Args:
            model: The model used to calculate the matching values.
            processor: The model used to obtain the embeddings.
            img_encoding: The image embedding.

        Returns:
            Tuple[str, str, float]: The best quantifier, noun, and the cosine similarity value.
        """

        text_list = []
        p = inflect.engine()  # Used to obtain plurals of words

        # Let's generate a list of sentences
        for quantificator in self.quantificators:
            for noun in self.nouns:
                text_list.append(Descriptor.get_sentence([quantificator + " " + noun]))
                text_list.append(Descriptor.get_sentence([quantificator + " " + p.plural(noun)]))

        # Find the best sentence
        position, value = self.get_text_descriptor(model, processor, text_list, img_encoding)

        # Get the noun and quantifier from the sentence
        noun = text_list[position][12:].split(" ")[-1]
        quantifier = text_list[position][12:][:-(1 + len(noun))]
        return quantifier, noun, value  # Quantifier, Noun, Value


    def add_word(self, model, processor, words_list, noun, quantifier, img_embedding, view = None):
        """
        Adds a word to the sentence in order to improve the cosine similarity.

        Args:
            model: The model used to calculate the matching values.
            processor: The model used to obtain the embeddings.
            words_list: The list of already used words.
            noun: The noun used to describe the object.
            quantifier: A quantifier used to describe the object.
            img_embedding: The embedding of the image.
            view: A view describing the image.

        Returns:
            Tuple[str, float]: The new word and the new cosine similarity value.
        """
        text_list = []
        # First generate a lsit with all the sentences
        for word in self.nouns_adjectives:
            text_list.append(Descriptor.get_sentence(words_list+[word],noun, quantifier))

        # Obtain the sentence that matches best the image embedding
        position, value = self.get_text_descriptor(model, processor, text_list, img_embedding)
        return self.nouns_adjectives[position], value

    def adjust_words(self, model, processor, words_list, noun, quantifier, img_embedding, value_max):
        """
        Adjusts the words in the sentence to maximize the cosine similarity score.

        Args:
            model: The model used to calculate the matching values.
            processor: The model used to obtain the embeddings.
            words_list: The list of already used words.
            noun: The noun used to describe the object.
            quantifier: A quantifier used to describe the object.
            img_embedding: The embedding of the image.
            value_max: The maximum cosine similarity value.

        Returns:
            float: The new maximum similarity value. Words are adjusted in place.
        """

        change_counter = 0
        checked = [] # Contains all the descriptor words already checked

        while len(checked) < len(words_list):
            # Pick an unchecked item
            item_num = torch.randint(len(words_list), (1,)).item()
            while item_num in checked:
                item_num = torch.randint(len(words_list), (1,)).item()

            # Check if changing the words improves the similarity score
            text_list = []
            base_word = words_list[item_num]
            for word in self.nouns_adjectives:
                words_list[item_num] = word
                text_list.append(Descriptor.get_sentence(words_list, noun, quantifier))
            position, value = self.get_text_descriptor(model, processor, text_list, img_embedding)

            # If it improves the similarity score update the word
            if value > value_max:
                value_max = value
                words_list[item_num] = self.nouns_adjectives[position]
                change_counter += 1
                if self.verbose: print("Change number ", change_counter, "                      ", end="\r")
                checked = [item_num] # Now all the other words can potentially change again
            else:
                words_list[item_num] = base_word
                checked.append(item_num)

        return value_max

    def decode_latents(self, latents):
        """
        Decode the latents into images
        Args:
            latents: latents of the images
        Returns: images
        """
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        return(imgs / 2 + 0.5).clamp(0, 1)

    def encode_imgs(self, imgs):
        """
        Encodes the images into lattent, reducing the shape.
        Args:
            imgs: images batchm, 3, 512,512
        Returns: latents of the images
        """
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='example', type=str)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--chk_every', type=int, default=500)
    parser.add_argument('--print_every', type=int, default=2500)
    parser.add_argument('--print_number', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--max_words_number', type=int, default=15)
    parser.add_argument('--noun', type=str)
    parser.add_argument('--quantifier', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--load_numb', type=int, default=-1)
    opt = parser.parse_args()

    if opt.verbose: print("Iteration muber: ", opt.iterations, " saving every ", opt.chk_every, " and starting from ", opt.load_numb)
    d = Descriptor(opt.image_path, workspace = opt.workspace, verbose = opt.verbose, num_words=opt.max_words_number)
    if not os.path.exists("./" + opt.workspace + "/descriptor/status.txt"):
        d.get_base_sentence( opt.noun, opt.quantifier)

    d.fit_description(opt.chk_every, opt.print_every, opt.iterations, samples = opt.print_number, load=opt.load_numb)

