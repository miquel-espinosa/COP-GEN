from pathlib import Path
import numpy as np
import rasterio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class TiffMetricEvaluator:
    def __init__(self, gt_dir, pred_dir, crop_size=256, output_name="metrics.txt"):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.crop_size = crop_size
        self.output_name = output_name
        self.rmse_list = []
        self.ssim_list = []

    def _load_tif(self, path):
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        return arr

    def _center_crop(self, arr, size):
        _, h, w = arr.shape
        ch, cw = size
        top = (h - ch) // 2
        left = (w - cw) // 2
        return arr[:, top:top + ch, left:left + cw]

    def _compute_rmse(self, a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    def _compute_ssim(self, a, b):
        if a.shape[0] == 1:
            range_val = b.max() - b.min()
            if range_val == 0:
                return np.nan
            return ssim(a[0], b[0], data_range=range_val)
        else:
            values = []
            for i in range(a.shape[0]):
                range_val = b[i].max() - b[i].min()
                if range_val == 0:
                    values.append(np.nan)
                else:
                    values.append(ssim(a[i], b[i], data_range=range_val))
            return np.nanmean(values)

    def evaluate(self):
        pred_files = sorted(self.pred_dir.glob("*.tif"))
        if not pred_files:
            print(f"No predicted TIFFs found in {self.pred_dir}")
            return

        for pred_file in tqdm(pred_files, desc="Evaluating"):
            tile_name = pred_file.stem
            gt_file = self.gt_dir / f"{tile_name}.tif"

            if not gt_file.exists():
                print(f"Missing ground truth: {gt_file}")
                continue

            pred = self._load_tif(pred_file)
            gt = self._load_tif(gt_file)

            if pred.shape != gt.shape:
                try:
                    gt = self._center_crop(gt, (pred.shape[1], pred.shape[2]))
                except Exception as e:
                    print(f"Failed to crop {tile_name}: {e}")
                    continue

            if pred.shape != gt.shape:
                print(f"Still mismatched shape after cropping {tile_name}: {pred.shape} vs {gt.shape}")
                continue

            self.rmse_list.append(self._compute_rmse(gt, pred))
            self.ssim_list.append(self._compute_ssim(gt, pred))

        self._report()

    def _report(self):
        mean_rmse = np.mean(self.rmse_list) if self.rmse_list else float("nan")
        mean_ssim = np.mean(self.ssim_list) if self.ssim_list else float("nan")

        print(f"\nEvaluation complete:")
        print(f"Average RMSE:  {mean_rmse:.4f}")
        print(f"Average SSIM:  {mean_ssim:.4f}")

        out_path = self.pred_dir / self.output_name
        with open(out_path, 'w') as f:
            f.write(f"Average RMSE: {mean_rmse:.4f}\n")
            f.write(f"Average SSIM: {mean_ssim:.4f}\n")
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    input = "DEM"
    output = "DEM_from_S1RTC_LULC_S2L1C_S2L2A"

    gt_dir_path = f"/home/egm/Data/Projects/CopGen/data/input/{input}"
    pred_dir_path = f"/home/egm/Data/Projects/CopGen/data/output/{output}"

    evaluator = TiffMetricEvaluator(gt_dir_path, pred_dir_path)
    evaluator.evaluate()


    # TODO so what if the resolution are different?
