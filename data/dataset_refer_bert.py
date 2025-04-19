import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import random
from bert.tokenization_bert import BertTokenizer
from refer.refer import REFER


class ReferDataset(data.Dataset):
    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.dataset = args.dataset
        self.max_tokens = 20
        self.eval_mode = eval_mode

        print('Preparing dataset ......')
        print(args.dataset, split)

        if args.dataset in ['LandRef']:
            self.ref_ids = []
            for ref in self.refer.data['refs'][split]:
                self.ref_ids.append(ref['ref_id'])

            img_ids = list(set(ref['image_id'] for ref in self.refer.data['refs'][split]))
            self.imgs = []
            for img_id in img_ids:
                self.imgs.append({
                    'id': img_id,
                    'file_name': f"{img_id}.tif"
                })
        else:
            self.ref_ids = self.refer.getRefIds(split=self.split)
            img_ids = self.refer.getImgIds(self.ref_ids)
            all_imgs = self.refer.Imgs
            self.imgs = list(all_imgs[i] for i in img_ids)

        num_images_to_mask = int(len(self.ref_ids) * 0.2)
        self.images_to_mask = random.sample(self.ref_ids, num_images_to_mask)

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        for r in self.ref_ids:
            if args.dataset in ['LandRef']:
                ref = self.refer.Refs[r]
                sentences_raw = [s['sent'] for s in ref['sentences']]
            else:
                ref = self.refer.Refs[r]
                sentences_raw = [el['raw'] for el in ref['sentences']]

            sentences_for_ref = []
            attentions_for_ref = []

            for sentence_raw in sentences_raw:
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                input_ids = input_ids[:self.max_tokens]  # 截断

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1] * len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def __len__(self):
        return len(self.ref_ids)

    def add_random_boxes(self, img, min_num=20, max_num=60, size=32):
        h, w = size, size
        img = np.asarray(img).copy()
        img_size = img.shape[1]
        num = random.randint(min_num, max_num)
        for _ in range(num):
            y, x = random.randint(0, img_size - w), random.randint(0, img_size - h)
            img[y:y + h, x:x + w] = 0
        return Image.fromarray(img.astype('uint8'), 'RGB')

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]

        if self.dataset in ['LandRef']:
            ref = self.refer.Refs[this_ref_id]
            img_id = ref['image_id']
            img_info = next(img for img in self.imgs if img['id'] == img_id)

            img_path = os.path.join(self.refer.IMAGE_DIR, img_info['file_name'])
            img = Image.open(img_path).convert("RGB")

            mask_path = os.path.join(self.refer.ROOT_DIR, f'data/{self.dataset}/masks', f"{img_id}.tif")
            ref_mask = np.array(Image.open(mask_path))
            ref_mask = (ref_mask[:, :, 0] > 0).astype(np.uint8)

            annot = Image.fromarray(ref_mask, mode='P')
        else:
            this_img_id = self.refer.getImgIds(this_ref_id)
            this_img = self.refer.Imgs[this_img_id[0]]

            img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
            ref = self.refer.loadRefs(this_ref_id)
            ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1
            annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = self.add_random_boxes(img)

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask

    def get_category_id(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]
        return ref['category_id']