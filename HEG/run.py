#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#TODO fix the show_bb parsing problem


import sys

sys.path.insert(0, "modules/object_tracking/yolov5")

import argparse
import numpy as np
import cv2
from transformers import pipeline
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import json
from PIL import Image
from tqdm import tqdm
from collections import deque, defaultdict
from test import create_logger

from common.config_parser import get_config
from common.data_io import VideoDatasetLoader
from common.transforms import STTranTransform, YOLOv5Transform
from common.inference_utils import fill_sttran_entry_inference, detect_emotions
from common.image_processing import convert_annotation_frame_to_video,crop_roi
from common.model_utils import (
    bbox_pair_generation,
    concat_separated_head,
    construct_sliding_window,
    generate_sliding_window_mask,
)
from common.plot import draw_bboxes
from common.metrics_utils import generate_triplets_scores

from modules.object_tracking import HeadTracking, ObjectTracking
from modules.gaze_following import GazeFollowing
from modules.gaze_following.head_association import assign_human_head_frame
from modules.sthoip_transformer.sttran_gaze import STTranGazeCrossAttention
from modules.object_tracking import FeatureExtractionResNet101


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="path to source")
    parser.add_argument("--future", type=int, default=0, help="seconds in future")
    parser.add_argument("--cfg", type=str, default="configs", help="path to configs")
    parser.add_argument("--weights", type=str, default="weights", help="root folder for all pretrained weights")
    parser.add_argument("--imgsz", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--hoi-thres", type=float, default=0.25, help="threshold for HOI score")
    parser.add_argument("--out", type=str, default="output", help="output folder")
    parser.add_argument("--print", action="store_true", help="print HOIs")
    parser.add_argument("--show_bb",action='store_true',help="Whether to show the bounding boxes or not")
    parser.add_argument("--show_emotions",action='store_true',help="Whether to show the emotions or not")
    opt = parser.parse_args()
    return opt


@torch.no_grad()
def main(opt):
    logger = create_logger()
    source = Path(opt.source)
    if not source.exists():
        print(f"{source} does not exist, exit")
        return -1
    video_name = source.stem
    future = opt.future
    hoi_thres = opt.hoi_thres
    print_hois = opt.print
    cfg_path = Path(opt.cfg)
    # path for all model weights
    weight_path = Path(opt.weights)
    yolo_weight_path = weight_path / "yolov5"
    deepsort_weight_path = weight_path / "deep_sort"
    gaze_following_weight_path = weight_path / "detecting_attended" / "model_videoatttarget.pt"
    backbone_model_path = weight_path / "backbone"
    sttran_word_vector_dir = weight_path / "semantic"
    model_path = weight_path / "sttrangaze" / f"f{future}_final.pt"
    # output files
    out = opt.out
    output_folder = Path(out) / video_name
    if not output_folder.exists():
        output_folder.mkdir()
    trace_file = output_folder / f"{video_name}_trace.json"
    gaze_file = output_folder / f"{video_name}_gaze.json"
    hoi_file = output_folder / f"{video_name}_hoi.txt"
    result_file = output_folder / f"{video_name}_result.json"
    result_video_file = output_folder / f"{video_name}_result.mp4"
    # model params
    imgsz = opt.imgsz
    gaze = "cross"
    global_token = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = get_config(str(cfg_path / "final" / f"eval_hyp_f{future}.yaml"))
    sampling_mode = cfg["sampling_mode"]
    dim_gaze_heatmap = cfg["dim_gaze_heatmap"]  # 64x64 always, dont care in test
    dim_transformer_ffn = cfg["dim_transformer_ffn"]
    sttran_enc_layer_num = cfg["sttran_enc_layer_num"]
    sttran_dec_layer_num = cfg["sttran_dec_layer_num"]
    sttran_sliding_window = cfg["sttran_sliding_window"]
    separate_head = cfg["separate_head"]  # always separate, dont care in test
    loss_type = cfg["loss_type"]  # only focal, dont care in test
    mlp_projection = cfg["mlp_projection"]  # MLP in input embedding
    sinusoidal_encoding = cfg["sinusoidal_encoding"]  # sinusoidal positional encoding

    # Emotion detection module
    # Use a pipeline as a high-level helper

    pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

    # Object Tracking module
    print(f"======================================")
    object_tracking_module = ObjectTracking(
        yolo_weights_path=str(yolo_weight_path / "vidor_yolov5l.pt"),
        deep_sort_model_dir=str(deepsort_weight_path),
        config_path=str(cfg_path / "object_tracking.yaml"),
        device=device,
    )
    yolov5_stride = object_tracking_module.yolov5_stride
    # Head Tracking and Gaze Following modules
    print(f"======================================")
    head_tracking_module = HeadTracking(
        crowd_human_weight_path=str(yolo_weight_path / "crowdhuman_yolov5m.pt"),
        deep_sort_model_dir=str(deepsort_weight_path),
        config_path=str(cfg_path / "object_tracking.yaml"),
        device=device,
    )
    print(f"======================================")
    gaze_following_module = GazeFollowing(
        weight_path=str(gaze_following_weight_path),
        config_path=str(cfg_path / "gaze_following.yaml"),
        device=device,
    )
    matching_iou_thres = 0.7
    matching_method = "hungarian"
    # Feature backbone
    print(f"======================================")
    feature_backbone = FeatureExtractionResNet101(backbone_model_path, download=True, finetune=False).to(device)
    feature_backbone.requires_grad_(False)
    feature_backbone.eval()
    print(f"Feature backbone loaded from {backbone_model_path}")
    # load available objects and interactions
    with Path("vidhoi_related/obj_categories.json").open("r") as f:
        object_classes = json.load(f)
    with Path("vidhoi_related/pred_categories.json").open("r") as f:
        interaction_classes = json.load(f)
    with Path("vidhoi_related/pred_split_categories.json").open("r") as f:
        temp_dict = json.load(f)
        spatial_class_idxes = temp_dict["spatial"]
        action_class_idxes = temp_dict["action"]
    num_object_classes = len(object_classes)
    num_interaction_classes = len(interaction_classes)
    num_spatial_classes = len(spatial_class_idxes)
    num_action_classes = len(action_class_idxes)
    num_interaction_classes_loss = num_interaction_classes
    # Transformer setup
    print(f"Transformer configs: {cfg}")
    loss_type_dict = {"spatial_head": "bce", "action_head": "bce"}
    separate_head_num = [num_spatial_classes, -1]
    separate_head_name = ["spatial_head", "action_head"]
    class_idxes_dict = {"spatial_head": spatial_class_idxes, "action_head": action_class_idxes}
    loss_gt_dict = {"spatial_head": "spatial_gt", "action_head": "action_gt"}
    sttran_gaze_model = STTranGazeCrossAttention(
        num_interaction_classes=num_interaction_classes_loss,
        obj_class_names=object_classes,
        spatial_layer_num=sttran_enc_layer_num,
        cross_layer_num=1,
        temporal_layer_num=sttran_dec_layer_num - 1,
        dim_transformer_ffn=dim_transformer_ffn,
        d_gaze=512,
        cross_sa=True,
        cross_ffn=False,
        global_token=global_token,
        mlp_projection=mlp_projection,
        sinusoidal_encoding=sinusoidal_encoding,
        dropout=0,
        word_vector_dir=sttran_word_vector_dir,
        sliding_window=sttran_sliding_window,
        separate_head=separate_head_num,
        separate_head_name=separate_head_name,
    )
    # load model weights
    sttran_gaze_model = sttran_gaze_model.to(device)
    incompatibles = sttran_gaze_model.load_state_dict(torch.load(model_path))
    sttran_gaze_model.eval()
    print(f"STTranGaze loaded. Incompatible keys {incompatibles}")

    # Load the video
    dataset = VideoDatasetLoader(
        source, transform=YOLOv5Transform(imgsz, yolov5_stride), additional_transform=STTranTransform(img_size=imgsz)
    )
    frame_num = dataset.frame_num
    print(f"Video {source} loaded with {frame_num} frames")
    # warmup tracker
    object_tracking_module.clear()
    frame, frame0, _, _, _ = next(iter(dataset))
    object_tracking_module.warmup(frame.to(device), frame0)
    print(f"Object tracker warmup finished.")

    # output video writer
    fourcc = "mp4v"
    fps, w, h = round(dataset.fps), frame0.shape[1], frame0.shape[0]  # not handle decimal fps
    video_writer = cv2.VideoWriter(str(result_video_file), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    # results for one video
    detection_dict = defaultdict(list)
    gaze_list = []
    hoi_list = []
    result_list = []
    # FIFO queues for sliding window
    frames_queue = deque(maxlen=sttran_sliding_window)
    frame_ids_queue = deque(maxlen=sttran_sliding_window)
    # iteration over the video, get object traces and human gazes
    # NOTE only store the detections and gazes in keyframes into files, but the video contains all frames
    hx_memory = {}
    print(f"============ Inference Start... ============")
    t = tqdm(enumerate(iter(dataset)), total=frame_num)
    # t = iter(dataset)
    for idx, batch in t:
        frame, frame0, _, _, meta_info = batch
        meta_info["original_shape"] = frame0.shape
        # object tracking
        bboxes, ids, labels, names, confs, _ = object_tracking_module.track_one(frame.to(device), frame0, draw=False)
        # draw bbox
        # logger.warning("IDS \n")
        # logger.critical(ids)
        # logger.warning("bboxes \n")
        # logger.critical(bboxes)
        # logger.warning("labels \n")
        # logger.critical(labels)
        # logger.warning("names \n")
        # logger.critical(names)
        # logger.warning("confs \n")
        # logger.critical(confs)
        if opt.show_bb and len(ids) > 0:
            frame_annotated = draw_bboxes(frame0.copy(), bboxes, ids, labels, names, confs)
        else:
            frame_annotated = frame0.copy()
        # get human bboxes
        human_idxes = np.array(labels) == 0
        human_bboxes = np.array(bboxes)[human_idxes]
        human_ids = np.array(ids)[human_idxes]
        # head detection
        h_bboxes, _, _, _, h_confs, _ = head_tracking_module.track_one(frame.to(device), frame0, draw=False)
        # human-head association
        head_bbox_dict = assign_human_head_frame(
            h_bboxes, h_confs, human_bboxes, human_ids, matching_iou_thres, matching_method
        )
        # emotion_dict = detect_emotions(frame_annotated,head_bbox_dict,pipe)

        
        # for _, coordinates in head_bbox_dict:
        #     if coordinates:
        #         x1,y1,x2,y2 = coordinates
        #         face = frame[x1:x2,y1:y2,:]
        #         face = cvtColor(face,COLOR_BGR2RGB)
        #         face = fromarray(face)
        #         emotion_dict = pipeline(face)
        #         dominant_emotion = emotion_dict[0]['label']
        #         returnable_dict[coordinates] = dominant_emotion

        # for i in head_bbox_dict:
        #     if len(head_bbox_dict[i]) != 0:
        #         cv2.rectangle(frame_annotated,(head_bbox_dict[i][0],head_bbox_dict[i][1]),(head_bbox_dict[i][2],head_bbox_dict[i][3]),(0,0,255),1)
        #         coordinates = head_bbox_dict[i]
        #         x1,y1,x2,y2 = coordinates
        #         face = crop_roi(frame_annotated.copy(),coordinates)
        #         face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        #         face = Image.fromarray(face)
        #         emotion_dict = pipe(face)
        #         dominant_emotion = emotion_dict[0]['label']
        #         returnable_dict[dominant_emotion] = coordinates
        
        # gaze following for each human
        frame_gaze_dict = {}
        for human_id, head_bbox in head_bbox_dict.items():
            # no head found for this human_id
            if len(head_bbox) == 0:
                frame_gaze_dict[int(human_id)] = []
                continue
            # check hidden state memory
            if human_id in hx_memory:
                hidden_state = hx_memory[human_id]
            else:
                hidden_state = None
            # gaze model forward
            heatmap, _, hx, _, _, frame_annotated = gaze_following_module.detect_one(
                frame0, head_bbox, hidden_state, draw=True, frame_to_draw=frame_annotated, id=human_id, arrow=True
            )
            # update hidden state memory
            hx_memory[human_id] = (hx[0].detach(), hx[1].detach())
            # process heatmap 64x64 (not include inout), store inout info separately
            frame_gaze_dict[int(human_id)] = heatmap.tolist()

        # store result for every second
        if idx % fps == 0:
            bboxes = [bbox.tolist() for bbox in bboxes]
            detection_dict["bboxes"].append(bboxes)
            detection_dict["ids"].append(ids)
            detection_dict["labels"].append(labels)
            detection_dict["confidences"].append(confs)
            detection_dict["frame_ids"].append(meta_info["frame_count"])
            gaze_list.append(frame_gaze_dict)
        # write video frame
        # video_writer.write(frame_annotated)
        # predict HOIs every second
        if idx % fps == 0:
            
            # Detect emotions every second
            final_emotion_dict = {}
            for i in head_bbox_dict:
                if len(head_bbox_dict[i]) != 0:
                    # cv2.rectangle(frame_annotated,(head_bbox_dict[i][0],head_bbox_dict[i][1]),(head_bbox_dict[i][2],head_bbox_dict[i][3]),(0,0,255),1)
                    face = crop_roi(frame_annotated.copy(),head_bbox_dict[i])
                    if face is not None:
                        face_numpy = face
                        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                        face = Image.fromarray(face)
                        emotion_dict = pipe(face)
                        dominant_emotion = emotion_dict[0]['label']
                        # w,h = cv2.getTextSize(dominant_emotion,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]
                        # x = round((head_bbox_dict[i][0] + head_bbox_dict[i][2] - w) / 2)
                        # y = head_bbox_dict[i][3] + h + 10
                        final_emotion_dict[dominant_emotion] = i
                        # logger.warning(f"x: {x}, y: {y}")

            # generate sliding window
            frames_queue.append(meta_info["additional"])
            frame_ids_queue.append(meta_info["frame_count"])
            sttran_frames = torch.cat(list(frames_queue)).to(device)
            det_bboxes = detection_dict["bboxes"][-sttran_sliding_window:]
            det_ids = detection_dict["ids"][-sttran_sliding_window:]
            det_labels = detection_dict["labels"][-sttran_sliding_window:]
            det_confidences = detection_dict["confidences"][-sttran_sliding_window:]
            bboxes, ids, pred_labels, confidences = convert_annotation_frame_to_video(
                det_bboxes, det_ids, det_labels, det_confidences
            )
            if len(bboxes) == 0:
                # no detection
                pair_idxes = []
                im_idxes = []
            else:
                # Generate human-object pairs
                pair_idxes, im_idxes = bbox_pair_generation(bboxes, pred_labels, 0)
            detected = {
                "bboxes": bboxes,
                "pred_labels": pred_labels,
                "ids": ids,
                "confidences": confidences,
                "pair_idxes": pair_idxes,
                "im_idxes": im_idxes,
            }
            # fill the entry with detections
            entry = fill_sttran_entry_inference(
                sttran_frames,
                detected,
                gaze_list[-sttran_sliding_window:],
                feature_backbone,
                meta_info,
                loss_type_dict,
                class_idxes_dict,
                loss_gt_dict,
                device,
                annotations=None,
                human_label=0,
            )
            
            windows = construct_sliding_window(entry, sampling_mode, sttran_sliding_window, 0, None, gt=False)
            entry, windows, windows_out, out_im_idxes, _ = generate_sliding_window_mask(entry, windows, None, "pair")

            # only do model forward if any valid window exists
            if len(windows) > 0:
                # everything to GPU
                entry["pair_idxes"] = entry["pair_idxes"].to(device)
                for i in range(len(entry["full_heatmaps"])):
                    entry["full_heatmaps"][i] = entry["full_heatmaps"][i].to(device)
                entry["pred_labels"] = entry["pred_labels"].to(device)
                entry["windows"] = windows.to(device)
                entry["windows_out"] = windows_out.to(device)

                # forward
                entry = sttran_gaze_model(entry)
                # sigmoid or softmax
                for head_name in loss_type_dict.keys():
                    if loss_type_dict[head_name] == "ce":
                        entry[head_name] = torch.softmax(entry[head_name], dim=-1)
                    else:
                        entry[head_name] = torch.sigmoid(entry[head_name])
                # in inference, length prediction may != length gt
                # len_preds = len(interactions_gt)
                len_preds = len(entry[list(loss_type_dict.keys())[0]])
                interaction_distribution = concat_separated_head(
                    entry, len_preds, loss_type_dict, class_idxes_dict, device, True
                )

            # process output
            frame_ids = list(frame_ids_queue)
            # window-wise result entry
            out_im_idx = len(frame_ids) - 1
            window_anno = {
                "video_name": video_name,  # video name
                "frame_id": frame_ids[out_im_idx],  # this frame id
            }
            if sampling_mode == "anticipation":
                if idx + future >= frame_num:
                    window_anno["future_frame_id"] = ""
                else:
                    window_anno["future_frame_id"] = f"{idx + fps * future:06d}"

            window_prediction = {
                "bboxes": [],
                "pred_labels": [],
                "confidences": [],
                "pair_idxes": [],
                "interaction_distribution": [],
            }
            # case 1, nothing detected in the full clip, result all []
            if len(entry["bboxes"]) == 0:
                pass
            else:
                det_out_idxes = entry["bboxes"][:, 0] == out_im_idx
                # case 2, nothing detected in this window, result all []
                if not det_out_idxes.any():
                    pass
                else:
                    # something detected, fill object detection results
                    # NOTE det_idx_offset is the first bbox index in this window
                    det_idx_offset = det_out_idxes.nonzero(as_tuple=True)[0][0]
                    window_prediction["bboxes"] = entry["bboxes"][det_out_idxes, 1:].numpy().tolist()
                    window_prediction["pred_labels"] = entry["pred_labels"][det_out_idxes].cpu().numpy().tolist()
                    window_prediction["confidences"] = entry["confidences"][det_out_idxes].numpy().tolist()
                    window_prediction["ids"] = torch.LongTensor(entry["ids"])[det_out_idxes].numpy().tolist()

                    pair_out_idxes = entry["im_idxes"] == out_im_idx
                    # case 3, no human-object pair detected (no human or no object), pair_idxes and distribution []
                    if not pair_out_idxes.any():
                        pass
                    else:
                        # case 4, have everything
                        pair_idxes = entry["pair_idxes"][pair_out_idxes] - det_idx_offset
                        # handle interaction distributions
                        window_prediction["pair_idxes"] = pair_idxes.cpu().numpy().tolist()
                        window_prediction["interaction_distribution"] = interaction_distribution.cpu().numpy().tolist()

            window_result = {**window_anno, **window_prediction}
            result_list.append(window_result)
            # print HOIs, only considering interaction scores
            triplets_scores = generate_triplets_scores(
                window_result["pair_idxes"],
                [1.0] * len(window_result["confidences"]),
                window_result["interaction_distribution"],
                multiply=True,
                top_k=100,
                thres=hoi_thres,
            )
            # s_hois = "-------------------------------\n"
            # s_hois += f"Frame {idx}/{frame_num}:\n"
            s_hois = []
            for score, idx_pair, interaction_pred in triplets_scores:
                interaction_name = interaction_classes[interaction_pred]
                if interaction_name in ['above','next_to','behind','in_front_of']:
                    continue
                subj_idx = window_result["pair_idxes"][idx_pair][0]
                subj_cls = window_result["pred_labels"][subj_idx]
                subj_name = object_classes[subj_cls]
                subj_id = window_result["ids"][subj_idx]
                obj_idx = window_result["pair_idxes"][idx_pair][1]
                obj_cls = window_result["pred_labels"][obj_idx]
                obj_name = object_classes[obj_cls]
                obj_id = window_result["ids"][obj_idx]
                # s_hois += f"{subj_name}{subj_id} - {interaction_name} - {obj_name}{obj_id}: {score}\n"
                s_hois.append((subj_name+str(subj_id),{'interaction':interaction_name,'doneon':obj_name,'conf':score}))
            hoi_list.append(s_hois)
            if print_hois:
                print(s_hois)
        name_id = [n+str(i) for n,i in zip(names,ids)]
        list_info = dict(zip(name_id,bboxes))
        color = (0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color_text = (255,255,255)
        tracker = {}
        x_tracker = {n:b[0] for n,b in list_info.items()}
        color_rects = [(72,167,233),(193,147,31),(38,52,211)]
        for hoi_record in s_hois:
            doer = hoi_record[0]
            interaction = hoi_record[1]['interaction']
            doneon = hoi_record[1]['doneon']
            conf = hoi_record[1]['conf']
            if doer in list_info:
                if doer in tracker.keys():
                    tracker[doer] += 1
                else:
                    tracker[doer] = 0
                x1,y1 = x_tracker[doer],list_info[doer][1]
                w, h = cv2.getTextSize(interaction+" "+doneon,font,font_scale,thickness)[0]
                cv2.rectangle(frame_annotated,(x1,y1-h-10), (x1+w,y1), color_rects[tracker[doer] % 3], -1)
                cv2.putText(frame_annotated,interaction+' '+doneon,(x_tracker[doer],y1-5),font,font_scale,color_text,1)
                x_tracker[doer] += w
        if opt.show_emotions:    
            for emotion,idx in final_emotion_dict.items():
                # logger.critical(head_bbox_dict)
                if idx in head_bbox_dict and len(head_bbox_dict[idx]) != 0:
                    bbox = head_bbox_dict[idx]
                    w,h = cv2.getTextSize(emotion,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]
                    x = round((bbox[0] + bbox[2] - w) / 2)
                    y = bbox[3] + h + 10
                    cv2.rectangle(frame_annotated,(x-5,y-h-5), (x+w+5,y+5), (143,83,7), -1)
                    cv2.putText(frame_annotated, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,255,255), 1, cv2.LINE_AA) 

        video_writer.write(frame_annotated)

    # release video writer
    video_writer.release()
    # store detections and gazes
    # with trace_file.open("w") as f:
    #     json.dump(detection_dict, f)
    # with gaze_file.open("w") as f:
    #     json.dump(gaze_list, f)
    # store HOIs
    


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
