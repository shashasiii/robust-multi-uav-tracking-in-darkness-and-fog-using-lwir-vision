# -*- coding: UTF-8 -*- #
"""
@filename:test_yolo_pt.py
@author:H.N. Deng
@time:2026-04-07
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 local test script")

    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to best.pt; default: <project_root>/weights/best.pt"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml; default: <project_root>/data/V4_yolo/data.yaml"
    )

    parser.add_argument(
        "--run_mode",
        type=str,
        default="predict_dir",
        choices=["val", "predict_image", "predict_dir", "compare_image"],
        help="Run mode"
    )

    parser.add_argument(
        "--source_img",
        type=str,
        default=None,
        help="Single image path for predict_image / compare_image"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Image folder path for predict_dir"
    )

    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0" or "cpu"')
    parser.add_argument("--batch", type=int, default=16, help="Batch size for val")
    parser.add_argument("--workers", type=int, default=4, help="Workers for val")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")

    parser.add_argument(
        "--output_name",
        type=str,
        default="yolo26n_test_run",
        help="Output run name prefix"
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Allow overwriting existing output directory"
    )

    return parser.parse_args()


def resolve_paths(args):
    project_root = Path(args.project_root)

    model_path = Path(args.model) if args.model else project_root / "weights" / "best.pt"
    data_yaml = Path(args.data) if args.data else project_root / "data" / "V4_yolo" / "data.yaml"

    source_img = Path(args.source_img) if args.source_img else project_root / "data" / "V4_yolo" / "test" / "images" / "sample.jpg"
    source_dir = Path(args.source_dir) if args.source_dir else project_root / "data" / "V4_yolo" / "test" / "images"

    output_project = project_root / "outputs"

    return project_root, model_path, data_yaml, source_img, source_dir, output_project


def check_paths(run_mode, model_path, data_yaml, source_img, source_dir):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if run_mode == "val":
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    elif run_mode in ["predict_image", "compare_image"]:
        if not source_img.exists():
            raise FileNotFoundError(f"Source image not found: {source_img}")

    elif run_mode == "predict_dir":
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")


def run_val(model, data_yaml, imgsz, batch, device, workers, conf, iou, output_project, output_name, exist_ok):
    print("\n===== Running evaluation on test split =====")
    print(f"Data yaml: {data_yaml}")

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        conf=conf,
        iou=iou,
        verbose=True,
        project=str(output_project),
        name=output_name + "_val",
        exist_ok=exist_ok,
    )

    print("\n===== Evaluation finished =====")
    print(metrics)
    print(f"Saved to: {output_project / (output_name + '_val')}")


def run_predict_image(model, source_img, imgsz, conf, iou, device, output_project, output_name, exist_ok):
    print("\n===== Running single-image prediction =====")
    print(f"Image: {source_img}")

    results = model.predict(
        source=str(source_img),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        show=False,
        project=str(output_project),
        name=output_name + "_image",
        exist_ok=exist_ok,
    )

    print("\n===== Single-image prediction finished =====")
    print(f"Results objects: {len(results)}")
    print(f"Saved to: {output_project / (output_name + '_image')}")


def run_predict_dir(model, source_dir, imgsz, conf, iou, device, output_project, output_name, exist_ok):
    print("\n===== Running folder prediction =====")
    print(f"Folder: {source_dir}")

    results = model.predict(
        source=str(source_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        show=False,
        project=str(output_project),
        name=output_name + "_dir",
        exist_ok=exist_ok,
    )

    print("\n===== Folder prediction finished =====")
    print(f"Results objects: {len(results)}")
    print(f"Saved to: {output_project / (output_name + '_dir')}")


def run_compare_image(model, source_img, imgsz, conf, iou, device, output_project, output_name, exist_ok):
    print("\n===== Running single-image comparison visualization =====")
    print(f"Image: {source_img}")

    # 1) 推理
    results = model.predict(
        source=str(source_img),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=True,          # 仍然保留，方便你单独拿预测图和txt
        save_txt=True,
        save_conf=True,
        show=False,
        project=str(output_project),
        name=output_name + "_compare_pred",
        exist_ok=exist_ok,
    )

    if len(results) == 0:
        raise RuntimeError("No prediction results returned.")

    result = results[0]

    # 2) 读取原图
    img_original = cv2.imread(str(source_img))
    if img_original is None:
        raise ValueError(f"Failed to read original image: {source_img}")

    # 3) 从内存中直接拿“带框预测图”
    # result.plot() 返回的是 RGB numpy array
    img_pred_rgb = result.plot()
    if img_pred_rgb is None:
        raise ValueError("Failed to generate plotted prediction image from result.plot().")

    # 转成 OpenCV 使用的 BGR
    img_pred = cv2.cvtColor(img_pred_rgb, cv2.COLOR_RGB2BGR)

    # 4) 保证两张图高度一致
    h1, w1 = img_original.shape[:2]
    h2, w2 = img_pred.shape[:2]
    target_h = max(h1, h2)

    if h1 != target_h:
        new_w1 = int(w1 * target_h / h1)
        img_original = cv2.resize(img_original, (new_w1, target_h))

    if h2 != target_h:
        new_w2 = int(w2 * target_h / h2)
        img_pred = cv2.resize(img_pred, (new_w2, target_h))

    # 5) 给两张图加标题条
    def add_title_bar(img, title, bar_height=50):
        h, w = img.shape[:2]
        bar = np.full((bar_height, w, 3), 255, dtype=np.uint8)
        cv2.putText(
            bar,
            title,
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return cv2.vconcat([bar, img])

    img_original_labeled = add_title_bar(img_original, "Original")
    img_pred_labeled = add_title_bar(img_pred, f"Prediction (conf={conf})")

    # 6) 左右拼接
    compare_img = cv2.hconcat([img_original_labeled, img_pred_labeled])

    # 7) 保存拼接结果
    compare_dir = output_project / (output_name + "_compare")
    compare_dir.mkdir(parents=True, exist_ok=True)

    compare_save_path = compare_dir / f"compare_{source_img.stem}.jpg"
    ok = cv2.imwrite(str(compare_save_path), compare_img)
    if not ok:
        raise IOError(f"Failed to save comparison image: {compare_save_path}")

    print("\n===== Comparison visualization finished =====")
    print(f"Detected boxes: {len(result.boxes) if result.boxes is not None else 0}")
    print(f"Original image: {source_img}")
    print(f"Comparison image saved to: {compare_save_path}")
    print(f"Prediction folder saved to: {output_project / (output_name + '_compare_pred')}")


def main():
    args = parse_args()

    (
        project_root,
        model_path,
        data_yaml,
        source_img,
        source_dir,
        output_project,
    ) = resolve_paths(args)

    print("===== YOLO test script =====")
    print(f"RUN_MODE = {args.run_mode}")
    print(f"PROJECT_ROOT = {project_root}")
    print(f"MODEL_PATH = {model_path}")
    print(f"CONF = {args.conf}")

    check_paths(args.run_mode, model_path, data_yaml, source_img, source_dir)

    model = YOLO(str(model_path))

    if args.run_mode == "val":
        run_val(
            model=model,
            data_yaml=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            output_project=output_project,
            output_name=args.output_name,
            exist_ok=args.exist_ok,
        )

    elif args.run_mode == "predict_image":
        run_predict_image(
            model=model,
            source_img=source_img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            output_project=output_project,
            output_name=args.output_name,
            exist_ok=args.exist_ok,
        )

    elif args.run_mode == "predict_dir":
        run_predict_dir(
            model=model,
            source_dir=source_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            output_project=output_project,
            output_name=args.output_name,
            exist_ok=args.exist_ok,
        )

    elif args.run_mode == "compare_image":
        run_compare_image(
            model=model,
            source_img=source_img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            output_project=output_project,
            output_name=args.output_name,
            exist_ok=args.exist_ok,
        )


if __name__ == "__main__":
    main()
