import numpy as np
import random
import torch
from torchvision import datasets, transforms
import os
import tqdm
from PIL import Image
import zipfile
import io
from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
import pickle
from transformers import AutoImageProcessor
import copy
import json
import glob
import json  

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
    
    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)


"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args, preprocessing="SAFSAR"):
        self.get_item_counter = 0
        self.debug_loader = args.debug_loader
        self.split = args.split
        self.query_per_class = args.query_per_class
        self.query_per_class_test = args.query_per_class_test
        self.data_dir = args.path
        self.seq_len = args.seq_len
        self.train = True
        self.tensor_transform = transforms.ToTensor()
        self.img_size = args.img_size

        self.annotation_path = args.traintestlist

        self.way=args.way
        self.shot=args.shot
        self.n_eval_steps=args.n_eval_steps

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

        # Change transform to custom ones if we are using SAFSAR
        if preprocessing == "SAFSAR":
            self.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            self.transform["train"] = self.custom_transform
            self.transform["test"] = self.custom_transform

        # Get complex names if they exists
        if os.path.exists(os.path.join(self.annotation_path, "classes_label_defn.json")):
            with open(os.path.join(self.annotation_path, "classes_label_defn.json"), 'r') as f:
                self.complex_classes_descriptions = json.load(f)
            for raw_label, complex_label in zip(self.class_folders, self.complex_classes_descriptions):
                assert raw_label.replace("_", "").replace(" ", "").lower() == complex_label["word"].replace("_", "").replace(" ", "").lower()
            self.complex_classes_descriptions = [x["cleaned_defn"] for x in self.complex_classes_descriptions]
        else:
            self.complex_classes_descriptions = copy.deepcopy(self.class_folders)  # if they not exists, use folder name

        ### Skeletons (optional) ###
        self.skeleton_root   = getattr(args, 'skeleton_root', None)
        # se non è passato, usa direttamente la root delle immagini
        # ### NEW ###
        if self.skeleton_root is None:                     
            self.skeleton_root = self.data_dir

        self.skeleton_format = getattr(args, 'skeleton_format', 'video_npy')  # "video_npy" | "frame_npy"
        # self.skeleton_J      = getattr(args, 'skeleton_J', 55)                # es. 30 per smpl+head_30
        # self.skeleton_J      = getattr(args, 'skeleton_J', 30)
        self.skeleton_J      = getattr(args, 'skeleton_J', 16)
        # self.skeleton_J      = getattr(args, 'skeleton_J', 48)                # es. 30 per smpl+head_30  
        # self.skeleton_C      = getattr(args, 'skeleton_C', 2)                 # 2: (x,y)  |  3: (x,y,conf)
        self.skeleton_stream = getattr(args, 'skeleton_stream', '3d')  # '3d' | '2d'
        if self.skeleton_stream == '3d':
            self.skeleton_C = 3  # (x,y,z)
        else:
            self.skeleton_C = getattr(args, 'skeleton_C', 2)

        self.use_skeleton    = (self.skeleton_root is not None) and os.path.exists(self.skeleton_root)

    def custom_transform(self, x):
        return [x for x in self.processor(x)["pixel_values"][0]]  # swapaxes(0, 2)

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []
            
        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        # video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)
    
    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        print(f"Loading data from {self.data_dir}")
        # load zipfile into memory
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            self.mem = open(zip_fn, 'rb').read()
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))
        else:
            self.zip = False

        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            
            # When using 'png' based datasets like kinetics, replace 'jpg' to 'png'
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))

            class_folders = list(set([x.split(os.sep)[-3] for x in dir_list if len(x.split(os.sep)) > 2]))
            class_folders.sort()
            self.class_folders = class_folders
            video_folders = list(set([x.split(os.sep)[-2] for x in dir_list if len(x.split(os.sep)) > 3]))
            video_folders.sort()
            self.video_folders = video_folders

            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}
            
            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()

            c = self.get_train_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:
            
                class_folder, video_folder, jpg = img_path.split(os.sep)[-3:]

                if video_folder != last_video_folder:
                    if len(insert_frames) >= self.seq_len:
                        c = self.get_train_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else:
                            pass
                    insert_frames = []
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]
               
                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id

            c = self.get_train_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
                c.add_vid(insert_frames, last_video_class)
        elif not os.path.exists(os.path.join(self.annotation_path, "preloaded_splits.pkl")) or self.debug_loader:
            class_folders = os.listdir(self.data_dir)
            class_folders.sort()
            self.class_folders = class_folders
            for class_folder in tqdm.tqdm(class_folders):
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                video_folders.sort()
                if self.debug_loader:
                    video_folders = video_folders[0:2]
                for video_folder in video_folders:
                    c = self.get_train_or_test_db(video_folder)
                    if c == None:
                        continue
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                    imgs = [x for x in imgs if x.endswith(".jpg")]  # ADDED BY ME
                    if len(imgs) < self.seq_len:
                        continue            
                    imgs.sort()
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                    paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].replace("frame_", "")))
                    class_id =  class_folders.index(class_folder)
                    c.add_vid(paths, class_id)
            if not self.debug_loader:  # save preloaded splits only if not debug
                with open(os.path.join(self.annotation_path, "preloaded_splits.pkl"), 'wb') as f:
                    pickle.dump((self.train_split, self.test_split), f)
        else:
            class_folders = os.listdir(self.data_dir)
            class_folders.sort()
            self.class_folders = class_folders
            with open(os.path.join(self.annotation_path, "preloaded_splits.pkl"), 'rb') as f:
                self.train_split, self.test_split = pickle.load(f)
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ return the current split being used """
    def get_train_or_test_db(self, split=None):
        if split is None:
            get_train_split = self.train
        else:
            split = split.lower() # ADDED BY ME
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
    
    """ load the paths of all videos in the train and test splits. """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.replace(' ', '_').lower() for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                
#                 if "kinetics" in self.args.path:
#                     data = [x[0:11] for x in data]
                
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        return 10000000
        if self.train:
            return 1000000
        else:
            return self.n_eval_steps
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i
            
    ### skeleton utils ###        
    def _split_parts(self, path):
        # gestisce sia path zip ("/") che fs (os.sep)
        return path.split(os.sep) if os.sep in path else path.split('/')

    def _img_to_class_video(self, img_path):
        parts = self._split_parts(img_path)
        return parts[-3], parts[-2]   # <classe>, <video>

    # def _skel_path_video_npz(self, class_folder, video_folder):
    #     # expected: <skeleton_root>/<class>/<video>/skeleton.npz
    #     cand = [
    #         os.path.join(self.skeleton_root, class_folder, video_folder, "skeleton.npz"),
    #         os.path.join(self.skeleton_root, class_folder, f"{video_folder}.npz"),
    #         os.path.join(self.skeleton_root, class_folder, video_folder, "skeleton.npy"),  # fallback
    #     ]
    #     for p in cand:
    #         if os.path.exists(p):
    #             return p
    #     return None
    def _skel_path_video_npz(self, class_folder, video_folder):

        ### change logic to find skeleton file associated to the video
        # Abilitare per avere scheletri Stefano
        # ora guarda prima nella cartella del video dentro skeleton_root
        base_dir = os.path.join(self.skeleton_root, class_folder, video_folder)
        cand = [
            #os.path.join(base_dir, "skeleton.npz"),
            #os.path.join(base_dir, "skeleton.npy"),
            os.path.join(base_dir, "poses.npy"),                         # ### NEW ###
            #os.path.join(self.skeleton_root, class_folder, f"{video_folder}.npz"),
            os.path.join(self.skeleton_root, class_folder, f"{video_folder}.npy")
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        # fallback: qualunque npz/npy presente nella cartella del video
        for ext in ("*.npz", "*.npy"):
            hits = glob.glob(os.path.join(base_dir, ext))
            if hits:
                return hits[0]
        return None

        # Abilitare per avere 55 keypoint da JSON normalizzati
        # Cerca **solo** i JSON normalizzati con subset: *_norm_subset.json
        # base_dir = os.path.join(self.skeleton_root, class_folder, video_folder)
        # hits = glob.glob(os.path.join(base_dir, "*_norm_subset.json"))
        # return hits[0] if hits else None


    def _skel_path_frame_npz(self, img_path):
        # expected: mirror della struttura immagini con .npz per frame
        rel = os.path.relpath(img_path, self.data_dir)
        base, _ = os.path.splitext(rel)
        p = os.path.join(self.skeleton_root, base + ".npz")
        return p if os.path.exists(p) else None

    def _empty_skeleton(self):
        return torch.zeros((self.seq_len, self.skeleton_J, self.skeleton_C), dtype=torch.float32)

    def _parse_meta(self, meta_obj):
        # l'export salva meta come json.dumps(...)
        if isinstance(meta_obj, (bytes, bytearray)):
            meta_str = meta_obj.decode('utf-8')
        else:
            meta_str = str(meta_obj)
        try:
            return json.loads(meta_str)
        except Exception:
            return {}

    def _stack_xy_conf(self, sk2d, conf2d):
        # sk2d: (T,J,2) ; conf2d: (T,J) or None  -> (T,J,C)
        if self.skeleton_C == 2:
            return sk2d.astype(np.float32)
        # C==3: terzo canale = conf (se disponibile), altrimenti 1.0
        if conf2d is None:
            conf2d = np.ones((sk2d.shape[0], sk2d.shape[1]), dtype=np.float32)
        conf2d = conf2d.astype(np.float32)[..., None]  # (T,J,1)
        return np.concatenate([sk2d.astype(np.float32), conf2d], axis=-1)  # (T,J,3)
    
    def _format_3d(self, sk3d):
        # sk3d: (T,J,3) o (J,3) -> (T,J,3)
        sk3d = np.array(sk3d, dtype=np.float32)
        if sk3d.ndim == 2:  # (J,3) -> (1,J,3)
            sk3d = sk3d[None, ...]
        return sk3d

    # def load_skeleton_seq(self, paths, idxs):
    #     """
    #     paths: lista di frame path per l'intero video
    #     idxs : indici selezionati da get_seq (len=seq_len)
    #     Ritorna torch.FloatTensor (seq_len, J, C)
    #     - se stream '3d' -> (x,y,z)
    #     - se stream '2d' -> (x,y) o (x,y,conf) a seconda di self.skeleton_C
    #     """
    #     if not self.use_skeleton:
    #         return self._empty_skeleton()

    #     J, C = self.skeleton_J, self.skeleton_C
    #     seq_len = len(idxs)
    #     use3d = (self.skeleton_stream == '3d')

    #     if self.skeleton_format == "video_npy":
    #         class_folder, video_folder = self._img_to_class_video(paths[0])
    #         p = self._skel_path_video_npz(class_folder, video_folder)
    #         if p is None:
    #             return self._empty_skeleton()

    #         data = np.load(p, allow_pickle=True)
    #         meta = self._parse_meta(data.get("meta", "{}"))

    #         # --- preferisci 3D se richiesto e disponibile ---
    #         if use3d and ("skeleton3d" in data):
    #             sk3d = self._format_3d(data["skeleton3d"])          # (T,J,3)
    #             T_avail = sk3d.shape[0]

    #             # prova match per nome frame dal meta (più robusto)
    #             sel_names = [os.path.basename(paths[i]) for i in idxs]
    #             take_idx = None
    #             if "frames" in meta and isinstance(meta["frames"], list) and len(meta["frames"]) >= 1:
    #                 fr_map = {os.path.basename(n): k for k, n in enumerate(meta["frames"])}
    #                 tmp = [fr_map.get(name, None) for name in sel_names]
    #                 if all(t is not None for t in tmp):
    #                     take_idx = tmp

    #             if take_idx is None:
    #                 if T_avail == len(paths):
    #                     take_idx = idxs
    #                 else:
    #                     take_idx = np.linspace(0, T_avail-1, num=seq_len, dtype=int).tolist()

    #             sk = sk3d[take_idx, :J, :3]                         # (seq_len,J,3)
    #             return torch.from_numpy(sk.astype(np.float32))

    #         # --- altrimenti usa 2D (xy [+ conf]) come prima ---
    #         sk2d = data.get("skeleton2d", None)                     # (T,J,2) o (J,2)
    #         conf = data.get("conf2d", None)                         # (T,J)   o (J,)
    #         if sk2d is None:
    #             return self._empty_skeleton()

    #         sk2d = np.array(sk2d)
    #         if sk2d.ndim == 2:                                      # (J,2) -> (1,J,2)
    #             sk2d = sk2d[None, ...]
    #         T_avail = sk2d.shape[0]

    #         sel_names = [os.path.basename(paths[i]) for i in idxs]
    #         take_idx = None
    #         if "frames" in meta and isinstance(meta["frames"], list) and len(meta["frames"]) >= 1:
    #             fr_map = {os.path.basename(n): k for k, n in enumerate(meta["frames"])}
    #             tmp = [fr_map.get(name, None) for name in sel_names]
    #             if all(t is not None for t in tmp):
    #                 take_idx = tmp
    #         if take_idx is None:
    #             if T_avail == len(paths):
    #                 take_idx = idxs
    #             else:
    #                 take_idx = np.linspace(0, T_avail-1, num=seq_len, dtype=int).tolist()

    #         sk2d = sk2d[take_idx, :J, :2]                            # (seq_len,J,2)
    #         if conf is not None:
    #             conf = np.array(conf)
    #             if conf.ndim == 1:                                   # (J,) -> (1,J)
    #                 conf = conf[None, ...]
    #             conf = conf[take_idx, :J]                            # (seq_len,J)

    #         sk = self._stack_xy_conf(sk2d, conf)                     # (seq_len,J,C)
    #         return torch.from_numpy(sk.astype(np.float32))

    #     elif self.skeleton_format == "frame_npy":
    #         sk_list = []
    #         for i in idxs:
    #             p = self._skel_path_frame_npz(paths[i])
    #             if p is None:
    #                 sk_list.append(np.zeros((J, C), dtype=np.float32))
    #                 continue
    #             data = np.load(p, allow_pickle=True)

    #             if use3d and ("skeleton3d" in data):
    #                 sk3d = np.array(data["skeleton3d"], dtype=np.float32)[:J, :3]  # (J,3)
    #                 sk_list.append(sk3d)
    #             else:
    #                 sk2d = data.get("skeleton2d", None)  # (J,2)
    #                 if sk2d is None:
    #                     sk_list.append(np.zeros((J, C), dtype=np.float32))
    #                     continue
    #                 sk2d = np.array(sk2d, dtype=np.float32)[:J, :2]
    #                 conf = data.get("conf2d", None)
    #                 if conf is not None:
    #                     conf = np.array(conf, dtype=np.float32)[:J]
    #                     sk = self._stack_xy_conf(sk2d[None, ...], conf[None, ...])[0]
    #                 else:
    #                     sk = self._stack_xy_conf(sk2d[None, ...], None)[0]
    #                 sk_list.append(sk)

    #         sk = np.stack(sk_list, axis=0)  # (seq_len,J,C)
    #         return torch.from_numpy(sk.astype(np.float32))

    #     else:
    #         # formato non supportato
    #         return self._empty_skeleton()
    #     ###
    def load_skeleton_seq(self, paths, idxs):
        """
        paths: lista di frame path per l'intero video
        idxs : indici selezionati da get_seq (len=seq_len)
        Ritorna torch.FloatTensor (seq_len, J, C)
        - se stream '3d' -> (x,y,z)
        - se stream '2d' -> (x,y) o (x,y,conf) a seconda di self.skeleton_C
        """
        import os, glob
        import numpy as np
        import torch

        if not self.use_skeleton:
            return self._empty_skeleton()

        J, C = self.skeleton_J, self.skeleton_C
        seq_len = len(idxs)
        use3d = (self.skeleton_stream == '3d')

        def _pick_indices(T_avail, seq_len, paths, idxs, meta_frames=None):
            """Prova ad allineare ai nomi frame (se disponibili), altrimenti match per lunghezza o risampling uniforme."""
            if meta_frames and isinstance(meta_frames, list) and len(meta_frames) >= 1:
                fr_map = {os.path.basename(n): k for k, n in enumerate(meta_frames)}
                sel_names = [os.path.basename(paths[i]) for i in idxs]
                tmp = [fr_map.get(n) for n in sel_names]
                if all(t is not None for t in tmp):
                    return tmp
            if T_avail == len(paths):
                return idxs
            # return np.linspace(0, T_avail - 1, num=seq_len, dtype=int).tolist()
            if len(paths) > 1:
                return [int(round(i * (T_avail - 1) / (len(paths) - 1))) for i in idxs]
            else:
                return [0] * seq_len

        def _find_video_skel_path():
            # """Cerca il file skeleton associato al video.
            # 1) usa la funzione esistente _skel_path_video_npz
            # 2) se non trovato, cerca nella cartella dei frame (poses.npy, skeleton.npy/npz, anche *.json)
            # """
            """Cerca il file skeleton associato al video **solo se è *_norm_subset.json**.
            1) prova _skel_path_video_npz
            2) altrimenti cerca nella cartella dei frame solo *_norm_subset.json
            """
            # tentativo con logica esistente (potrebbe puntare a skeleton_root esterno)
            class_folder, video_folder = self._img_to_class_video(paths[0])

            #abilitare per avere scheletri rtmpose3d
            p = self._skel_path_video_npz(class_folder, video_folder)
            if p and os.path.exists(p):
                return p
            # RTMPose3D
            p = self._skel_path_video_npz(class_folder, video_folder)
            if p and os.path.exists(p) and p.lower().endswith("_norm_subset.json"):
                return p

            # fallback: nella stessa cartella dei frame
            base_dir = os.path.dirname(paths[0])
            #abilitare per avere scheletri Stefano
            candidates = [
                os.path.join(base_dir, "poses.npy")
                #os.path.join(base_dir, "skeleton.npy"),
                #os.path.join(base_dir, "skeleton.npz"),
                #os.path.join(base_dir, "skeleton.json"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
            # # qualunque npz/npy nella cartella
            hits = glob.glob(os.path.join(base_dir, "*.npz")) + glob.glob(os.path.join(base_dir, "*.npy"))
            # qualunque npz/npy/json nella cartella
            hits = (
                glob.glob(os.path.join(base_dir, "*.npz"))
                + glob.glob(os.path.join(base_dir, "*.npy"))
                + glob.glob(os.path.join(base_dir, "*.json"))
                )
            return hits[0] if hits else None

            # Abilitare per avere 55 keypoint da JSON normalizzati
            # hits = glob.glob(os.path.join(base_dir, "*_norm_subset.json"))
            # return hits[0] if hits else None

        if self.skeleton_format == "video_npy":
            p = _find_video_skel_path()
            if p is None:
                return self._empty_skeleton()

            #data = np.load(p, allow_pickle=True)
            # --- NUOVO: supporto JSON con subset 55 keypoint ---
            if p.lower().endswith(".json"):
                # usa solo i file normalizzati con subset
                if not p.lower().endswith("_norm_subset.json"):
                    return self._empty_skeleton()
                with open(p, "r") as f:
                    jd = json.load(f)

                # 1) subset di 55 indici (ordine da rispettare)
                # keep_ids = jd.get("meta_info", {}).get("subset", {}).get("keep_ids", None)
                # if not keep_ids:
                #     return self._empty_skeleton()
                # keep_ids = np.array(keep_ids, dtype=int)   # 55 indici
                # Abiliatare per usare json con 55 keypoint
                # keep_ids = jd.get("meta_info", {}).get("subset", {}).get("keep_ids", None)
                # Abiliatare per usare json con 48 keypoint su norm subset
                keep_ids = np.array(
                    [5, 6, 7, 8, 9, 10, ] + list(range(91, 133)), dtype=int
                )
                J = self.skeleton_J 
                if keep_ids is not None:
                    keep_ids = np.array(keep_ids, dtype=int)

                # 2) costruisci (T, 55, 3) dai 133×3
                frames = jd.get("instance_info", [])
                seq_all = []
                for fr in frames:
                    insts = fr.get("instances", [])
                    if not insts:
                        # frame senza persona -> zeri
                        seq_all.append(np.zeros((len(keep_ids), 3), dtype=np.float32))
                        continue
                    kps = np.array(insts[0]["keypoints"], dtype=np.float32)  # (133,3)
                    if kps.ndim != 2 or kps.shape[1] < 3:
                        seq_all.append(np.zeros((len(keep_ids), 3), dtype=np.float32))
                        continue
                    #kp55 = kps[keep_ids, :3]                                 # (55,3)
                    num_kp = kps.shape[0]
                    # Se il JSON contiene 133 kp, applico il subset.
                    # Se contiene già 55 (file *_norm_subset.json), li uso così come sono.
                    #if keep_ids is not None and num_kp >= (max(keep_ids) + 1):
                    if (keep_ids is not None) and (num_kp > int(np.max(keep_ids))):
                        kp_sel = kps[keep_ids, :3]                           # (55,3) da 133
                    #elif num_kp == len(keep_ids) or num_kp == J:
                    elif (num_kp == J) or (keep_ids is not None and num_kp == len(keep_ids)):
                        kp_sel = kps[:J, :3]                                 # già subsettati (55,3)
                    else:
                        # fallback robusto: tronca/padda a J
                        kp_sel = np.zeros((J, 3), dtype=np.float32)
                        take = min(J, num_kp)
                        kp_sel[:take] = kps[:take, :3]
                    seq_all.append(kp_sel)
                    #seq_all.append(kp55)
                if len(seq_all) == 0:
                    return self._empty_skeleton()
                sk3d = np.stack(seq_all, axis=0)                             # (T,55,3)

                # 3) allineamento temporale ai frame immagine
                T_avail = sk3d.shape[0]
                take_idx = _pick_indices(T_avail, seq_len, paths, idxs, meta_frames=None)
                sk = sk3d[take_idx, :J, :3]                                  # (seq_len, J, 3)
                # padding se J (config) > 55
                if sk.shape[1] < J:
                    padJ = np.zeros((sk.shape[0], J - sk.shape[1], 3), dtype=np.float32)
                    sk = np.concatenate([sk, padJ], axis=1)
                return torch.from_numpy(sk.astype(np.float32))

            # --- FINE ramo JSON; prosegui con npz/npy standard ---
            data = np.load(p, allow_pickle=True)



            # --- Caso .npz con chiavi note ---
            if isinstance(data, np.lib.npyio.NpzFile):
                # parse meta se presente
                meta_raw = data.get("meta", "{}")
                meta = self._parse_meta(meta_raw) if meta_raw is not None else {}

                # preferisci 3D se richiesto e disponibile
                if use3d and ("skeleton3d" in data.files or "skeleton3d" in data):
                    # self._format_3d deve restituire (T,J,3)
                    sk3d = self._format_3d(data["skeleton3d"])
                    T_avail = sk3d.shape[0]
                    take_idx = _pick_indices(T_avail, seq_len, paths, idxs, meta.get("frames"))
                    sk = sk3d[take_idx, :J, :3]
                    if sk.shape[1] < J:
                        padJ = np.zeros((sk.shape[0], J - sk.shape[1], 3), dtype=np.float32)
                        sk = np.concatenate([sk, padJ], axis=1)
                    return torch.from_numpy(sk.astype(np.float32))

                # altrimenti usa 2D (+conf se disponibile)
                sk2d = data.get("skeleton2d", None)
                if sk2d is None:
                    return self._empty_skeleton()
                sk2d = np.array(sk2d)
                if sk2d.ndim == 2:  # (J,2) -> (1,J,2)
                    sk2d = sk2d[None, ...]
                T_avail = sk2d.shape[0]

                take_idx = _pick_indices(T_avail, seq_len, paths, idxs, meta.get("frames"))
                sk2d = sk2d[take_idx, :J, :2]  # (seq_len,J,2)
                if sk2d.shape[1] < J:
                    padJ = np.zeros((sk2d.shape[0], J - sk2d.shape[1], 2), dtype=np.float32)
                    sk2d = np.concatenate([sk2d, padJ], axis=1)

                conf = data.get("conf2d", None)
                if conf is not None:
                    conf = np.array(conf)
                    if conf.ndim == 1:
                        conf = conf[None, ...]
                    conf = conf[take_idx, :J]  # (seq_len,J)
                sk = self._stack_xy_conf(sk2d, conf)  # (seq_len,J,C)
                return torch.from_numpy(sk.astype(np.float32))

            # --- Caso .npy semplice: array nudo ---
            arr = np.array(data)
            # normalizza a float32
            arr = arr.astype(np.float32, copy=False)

            # normalizza shape a (T,J,C)
            if arr.ndim == 2:             # (J,C) -> (1,J,C)
                arr = arr[None, ...]
            elif arr.ndim != 3:
                return self._empty_skeleton()

            T_avail, J_avail, C_avail = arr.shape

            # selezione temporale coerente con la sequenza immagini
            take_idx = _pick_indices(T_avail, seq_len, paths, idxs, None)

            if use3d:
                # usa i primi 3 canali come (x,y,z); se mancanti, pad con 1.0
                if C_avail < 3:
                    pad = np.ones((T_avail, J_avail, 3 - C_avail), dtype=np.float32)
                    arr3 = np.concatenate([arr, pad], axis=-1)
                else:
                    arr3 = arr[..., :3]

                # ADDED TO SELECT THE SUBSET OF 16 JOINTS FROM STE'S SKTELETONS
                keep_ids = np.array([0, 3, 6, 9, 12, 13, 14, 15,
                                    16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int64)
                # se il file ha abbastanza joint, seleziona; altrimenti fallback robusto
                if J_avail >= int(keep_ids.max()) + 1:
                    arr3 = arr3[:, keep_ids, :]        # (T, 16, 3)
                    J_avail = arr3.shape[1]
                else:
                    # fallback: prendi i primi J_avail e pad a J=16
                    pass


                sk = arr3[take_idx, :min(J, J_avail), :3]
                # se J_avail < J, pad joint extra con zeri
                if J_avail < J:
                    padJ = np.zeros((len(take_idx), J - J_avail, 3), dtype=np.float32)
                    sk = np.concatenate([sk, padJ], axis=1)
                return torch.from_numpy(sk.astype(np.float32))
            else:
                # 2D: prendi (x,y)
                if C_avail < 2:
                    return self._empty_skeleton()
                sk2d = arr[take_idx, :min(J, J_avail), :2]
                # pad joint se necessario
                if J_avail < J:
                    padJ = np.zeros((len(take_idx), J - J_avail, 2), dtype=np.float32)
                    sk2d = np.concatenate([sk2d, padJ], axis=1)
                conf = None
                if C == 3:
                    conf = np.ones((len(take_idx), J), dtype=np.float32)
                    if J_avail < J:
                        # già allineato a J con pad, conf a 1 ovunque
                        pass
                sk = self._stack_xy_conf(sk2d, conf)  # (seq_len,J,C)
                return torch.from_numpy(sk.astype(np.float32))

        elif self.skeleton_format == "frame_npy":
            # comportamento precedente, ma robusto su npz/npy
            sk_list = []
            for i in idxs:
                p = self._skel_path_frame_npz(paths[i])
                if p is None:
                    sk_list.append(np.zeros((J, C), dtype=np.float32))
                    continue
                data = np.load(p, allow_pickle=True)

                # npz con chiavi note
                if isinstance(data, np.lib.npyio.NpzFile):
                    if use3d and ("skeleton3d" in data):
                        sk3d = np.array(data["skeleton3d"], dtype=np.float32)[:J, :3]
                        sk_list.append(sk3d)
                        continue
                    sk2d = data.get("skeleton2d", None)
                    if sk2d is None:
                        sk_list.append(np.zeros((J, C), dtype=np.float32))
                        continue
                    sk2d = np.array(sk2d, dtype=np.float32)[:J, :2]
                    conf = data.get("conf2d", None)
                    if conf is not None:
                        conf = np.array(conf, dtype=np.float32)[:J]
                        sk = self._stack_xy_conf(sk2d[None, ...], conf[None, ...])[0]
                    else:
                        sk = self._stack_xy_conf(sk2d[None, ...], None)[0]
                    sk_list.append(sk)
                else:
                    # .npy semplice per frame: atteso (J,C) o (J,2/3)
                    arr = np.array(data, dtype=np.float32)
                    if arr.ndim != 2:
                        sk_list.append(np.zeros((J, C), dtype=np.float32))
                        continue
                    arr = arr[:J, :min(arr.shape[1], 3)]
                    if use3d:
                        # pad a 3 canali se servono
                        if arr.shape[1] < 3:
                            pad = np.ones((arr.shape[0], 3 - arr.shape[1]), dtype=np.float32)
                            arr = np.concatenate([arr, pad], axis=-1)
                        sk_list.append(arr[:, :3])
                    else:
                        if arr.shape[1] < 2:
                            sk_list.append(np.zeros((J, C), dtype=np.float32))
                            continue
                        sk2d = arr[:, :2]
                        conf = None
                        if C == 3:
                            conf = np.ones((J,), dtype=np.float32)
                            sk = self._stack_xy_conf(sk2d[None, ...], conf[None, ...])[0]
                        else:
                            sk = self._stack_xy_conf(sk2d[None, ...], None)[0]
                        sk_list.append(sk)

            sk = np.stack(sk_list, axis=0)  # (seq_len,J,C) o (seq_len,J,3) per 3d
            return torch.from_numpy(sk.astype(np.float32))

        else:
            # formato non supportato
            return self._empty_skeleton()


    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        n_frames = len(paths)
        if n_frames == self.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames-1 -excess_pad, n_frames-1)
            else:
                start = 1
                end = n_frames - 2
    
            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]
        # print(f"opening {paths[0]}")
        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs, vid_id
    
    ### skeleton version of get_seq ###
    def get_seq_with_skeleton(self, label, idx: int = -1):
        """Come get_seq, ma allinea gli idx ai frame salvati nello skeleton (se disponibile)
        e restituisce anche lo skeleton (seq_len, J, C)."""
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        n_frames = len(paths)

        # 1) sampling come in get_seq (stessa logica)
        if n_frames == self.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames - 1 - excess_pad, n_frames - 1)
            else:
                start = 1
                end = n_frames - 2

            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0

            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            if self.seq_len == 1:
                idxs = [random.randint(start, end - 1)]

        # 2) se abbiamo skeleton video_npy con meta["frames"], sovrascrivi idxs
        if self.use_skeleton and self.skeleton_format == "video_npy":
            try:
                class_folder, video_folder = self._img_to_class_video(paths[0])
                p = self._skel_path_video_npz(class_folder, video_folder)
                if p is not None:
                    data = np.load(p, allow_pickle=True)
                    meta = self._parse_meta(data.get("meta", "{}"))
                    frames_meta = meta.get("frames", None)
                    if isinstance(frames_meta, list) and len(frames_meta) == self.seq_len:
                        name_to_idx = {os.path.basename(pp): ii for ii, pp in enumerate(paths)}
                        idxs_meta = [name_to_idx.get(os.path.basename(nm)) for nm in frames_meta]
                        if all(ii is not None for ii in idxs_meta):
                            idxs = idxs_meta  # usa esattamente i 16 frame salvati nello skeleton
            except Exception:
                # in caso di problemi, mantieni idxs originali
                pass

        # 3) carica immagini per gli idxs finali
        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            transform = self.transform["train"] if self.train else self.transform["test"]
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)

        # 4) carica skeleton allineato (usa gli stessi idxs)
        skels = self.load_skeleton_seq(paths, idxs)  # (seq_len, J, C) -> per te C=3 (x,y,z)

        return imgs, skels, vid_id
        ###



    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)  # this works without replacement

        if self.train:
            n_queries = self.query_per_class
        else:
            n_queries = self.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []
        unknown_set = []
        unknown_labels = []

        ### skeleton lists ###
        support_skeletons = []   # NEW
        target_skeletons  = []   # NEW
        unknown_skeletons = []   # NEW

        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.shot + n_queries)

            for idx in idxs[0:self.shot]:
                # vid, vid_id = self.get_seq(bc, idx)
                # support_set.append(vid)
                # support_labels.append(bl)
                vid, skel, vid_id = self.get_seq_with_skeleton(bc, idx)  # NEW
                support_set.append(vid)
                support_skeletons.append(skel)                            # NEW
                support_labels.append(bl)

            for idx in idxs[self.shot:]:
                # vid, vid_id = self.get_seq(bc, idx)
                # target_set.append(vid)
                # target_labels.append(bl)
                # real_target_labels.append(bc)
                vid, skel, vid_id = self.get_seq_with_skeleton(bc, idx)  # NEW
                target_set.append(vid)
                target_skeletons.append(skel)                             # NEW
                target_labels.append(bl)
                real_target_labels.append(bc)


        # Select unknown classes
        unknown_classes = [x for x in classes if x not in batch_classes]
        for _ in range(len(batch_classes)):
            for _ in range(n_queries):
                bc = random.choice(unknown_classes)
                n_total = c.get_num_videos_for_class(bc)
                idx = random.randint(0, n_total - 1)
                # vid, vid_id = self.get_seq(bc, idx)
                # unknown_set.append(vid)
                # unknown_labels.append(bc)
                vid, skel, vid_id = self.get_seq_with_skeleton(bc, idx)  # NEW
                unknown_set.append(vid)
                unknown_skeletons.append(skel)                            # NEW
                unknown_labels.append(bc)
        
        # s = list(zip(support_set, support_labels))
        # random.shuffle(s)
        # support_set, support_labels = zip(*s)

        # support
        s = list(zip(support_set, support_labels, support_skeletons))  
        random.shuffle(s)
        support_set, support_labels, support_skeletons = zip(*s)

        support_set       = torch.cat(support_set)          # (way*shot*seq_len, 3, H, W)
        support_skeletons = torch.cat(support_skeletons)    # (way*shot*seq_len, J, C)
        support_labels    = torch.FloatTensor(support_labels)
                
        # t = list(zip(target_set, target_labels, real_target_labels))
        # random.shuffle(t)
        # target_set, target_labels, real_target_labels = zip(*t)

        # target
        t = list(zip(target_set, target_labels, real_target_labels, target_skeletons))  # NEW
        random.shuffle(t)
        target_set, target_labels, real_target_labels, target_skeletons = zip(*t)       # NEW

        target_set       = torch.cat(target_set)            # (n_queries*seq_len, 3, H, W)
        target_skeletons = torch.cat(target_skeletons)      # (n_queries*seq_len, J, C)
        target_labels    = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)

        # unknown_set = torch.cat(unknown_set)
        # unknown_labels = torch.FloatTensor(unknown_labels)

        # unknown
        unknown_set       = torch.cat(unknown_set)          # (n_unknown*seq_len, 3, H, W)
        unknown_skeletons = torch.cat(unknown_skeletons)    # (n_unknown*seq_len, J, C)
        unknown_labels    = torch.FloatTensor(unknown_labels)
                
        # support_set = torch.cat(support_set)
        # target_set = torch.cat(target_set)
        # support_labels = torch.FloatTensor(support_labels)
        # target_labels = torch.FloatTensor(target_labels)
        # real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 

        return {"support_set":support_set, 
                "support_labels":support_labels, 
                "target_set":target_set, 
                "target_labels":target_labels, 
                "real_target_labels":real_target_labels, 
                "batch_class_list": batch_classes, 
                "unknown_set":unknown_set, 
                "unknown_labels":unknown_labels,
                "support_skeleton": support_skeletons,   # (way*shot*seq_len, J, C)
                "target_skeleton": target_skeletons,     # (n_queries*seq_len, J, C)
                "unknown_skeleton": unknown_skeletons,   # (n_unknown*seq_len, J, C)
            }


class HMDB:
    path = "/home/steb6/datasets/hmdb51/images"
    traintestlist = "splits/hmdb_ARN"
    seq_len = 16
    img_size = 224
    way = 5
    shot = 5
    query_per_class = 4
    split = 3
    debug_loader = False
    query_per_class_test = 1
    n_eval_steps = 10000

class SSv2:
    path = "/home/steb6/datasets/SSv2/images"
    traintestlist = "splits/ssv2_OTAM"
    seq_len = 16
    img_size = 224
    way = 5
    shot = 5
    query_per_class = 4
    split = 7
    debug_loader = False
    query_per_class_test = 1
    n_eval_steps = 10000

class UCF:
    path = "/home/steb6/datasets/ucf101/images"
    traintestlist = "splits/ucf_ARN"
    seq_len = 16
    img_size = 224
    way = 5
    shot = 5
    query_per_class = 4
    split = 3
    debug_loader = False
    query_per_class_test = 1
    n_eval_steps = 10000

    
if __name__ == "__main__":
    dataset = VideoDataset(SSv2())
