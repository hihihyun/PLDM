"""
Test script for Underwater Image Enhancement Diffusion Model
"""

import os
import torch
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import time

from models.main_model import UnderwaterEnhancementModel
from data.dataset import UnderwaterDataset
from utils import calculate_metrics, save_single_image, load_image
from config import get_config


class Tester:
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = UnderwaterEnhancementModel(model_path, self.device)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.ToTensor()
        ])
        
        print(f"Tester initialized. Device: {self.device}")
    
    def test_single_image(self, image_path, output_path=None, num_steps=50):
        """Test on a single image"""
        # Load and preprocess image
        image = load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Enhance image
        start_time = time.time()
        with torch.no_grad():
            enhanced = self.model.enhance_image(image_tensor, num_steps=num_steps)
        inference_time = time.time() - start_time
        
        # Convert back to PIL
        enhanced_pil = transforms.ToPILImage()(enhanced.squeeze(0).cpu())
        
        # Save result
        if output_path:
            enhanced_pil.save(output_path)
            print(f"Enhanced image saved to {output_path}")
        
        return enhanced_pil, inference_time
    
    def test_dataset(self, dataset_config, output_dir, num_steps=50):
        """Test on a dataset"""
        # Create dataset
        test_dataset = UnderwaterDataset(
            root_dir=dataset_config['data_root'],
            dataset_type=dataset_config['dataset_type'],
            split='test',
            img_size=self.config['img_size'],
            augment=False,
            preprocessing_type='none'
        )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        metrics_list = []
        total_time = 0
        
        print(f"Testing on {len(test_dataset)} images...")
        
        for idx in tqdm(range(len(test_dataset))):
            sample = test_dataset[idx]
            degraded = sample['degraded'].unsqueeze(0)
            filename = sample['filename']
            
            # Enhance image
            start_time = time.time()
            with torch.no_grad():
                enhanced = self.model.enhance_image(degraded, num_steps=num_steps)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Save enhanced image
            enhanced_pil = transforms.ToPILImage()(enhanced.squeeze(0).cpu())
            output_path = output_dir / f"enhanced_{filename}"
            enhanced_pil.save(output_path)
            
            # Calculate metrics if ground truth is available
            if 'enhanced' in sample:
                ground_truth = sample['enhanced']
                metrics = calculate_metrics(enhanced.squeeze(0), ground_truth)
                metrics['filename'] = filename
                metrics['inference_time'] = inference_time
                metrics_list.append(metrics)
        
        # Save metrics
        if metrics_list:
            self.save_metrics(metrics_list, output_dir / 'metrics.json')
        
        avg_time = total_time / len(test_dataset)
        print(f"Average inference time: {avg_time:.3f}s")
        
        return metrics_list
    
    def test_batch(self, image_paths, output_dir, batch_size=4, num_steps=50):
        """Test on a batch of images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_time = 0
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load batch
            batch_images = []
            for path in batch_paths:
                image = load_image(path)
                image_tensor = self.transform(image)
                batch_images.append(image_tensor)
            
            batch_tensor = torch.stack(batch_images)
            
            # Enhance batch
            start_time = time.time()
            with torch.no_grad():
                enhanced_batch = self.model.enhance_batch(batch_tensor, num_steps=num_steps)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Save results
            for j, (enhanced, path) in enumerate(zip(enhanced_batch, batch_paths)):
                enhanced_pil = transforms.ToPILImage()(enhanced.cpu())
                filename = Path(path).stem
                output_path = output_dir / f"enhanced_{filename}.png"
                enhanced_pil.save(output_path)
        
        avg_time = total_time / len(image_paths)
        print(f"Average inference time per image: {avg_time:.3f}s")
    
    def benchmark_speed(self, image_size=(256, 256), num_trials=100, num_steps_list=[10, 20, 50, 100]):
        """Benchmark inference speed"""
        print("Benchmarking inference speed...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *image_size).to(self.device)
        
        results = {}
        
        for num_steps in num_steps_list:
            print(f"Testing with {num_steps} steps...")
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model.enhance_image(dummy_input, num_steps=num_steps)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_trials):
                with torch.no_grad():
                    _ = self.model.enhance_image(dummy_input, num_steps=num_steps)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_trials
            results[num_steps] = avg_time
            
            print(f"  Average time: {avg_time:.3f}s")
        
        return results
    
    def save_metrics(self, metrics_list, output_path):
        """Save metrics to JSON file"""
        # Calculate average metrics
        if not metrics_list:
            return
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key not in ['filename', 'inference_time']:
                values = [m[key] for m in metrics_list if key in m]
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
        
        # Add timing info
        avg_metrics['avg_inference_time'] = np.mean([m['inference_time'] for m in metrics_list])
        
        # Save detailed and average metrics
        output_data = {
            'average_metrics': avg_metrics,
            'detailed_metrics': metrics_list
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Metrics saved to {output_path}")
        print("Average metrics:")
        for key, value in avg_metrics.items():
            if not key.startswith('std_'):
                print(f"  {key}: {value:.4f}")
    
    def qualitative_evaluation(self, image_paths, output_dir, num_steps=50):
        """Generate qualitative results for visual inspection"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(tqdm(image_paths)):
            # Load original image
            original = load_image(image_path)
            original_tensor = self.transform(original).unsqueeze(0)
            
            # Generate enhancement
            enhanced, inference_time = self.test_single_image(image_path, num_steps=num_steps)
            
            # Create comparison image
            comparison = Image.new('RGB', (original.width * 2, original.height))
            comparison.paste(original, (0, 0))
            comparison.paste(enhanced.resize(original.size), (original.width, 0))
            
            # Save comparison
            filename = Path(image_path).stem
            comparison_path = output_dir / f"comparison_{filename}.png"
            comparison.save(comparison_path)
            
            # Save individual enhanced image
            enhanced_path = output_dir / f"enhanced_{filename}.png"
            enhanced.save(enhanced_path)
        
        print(f"Qualitative results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test Underwater Image Enhancement Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['single', 'dataset', 'batch', 'benchmark'], 
                       default='single', help='Test mode')
    
    # Single image mode
    parser.add_argument('--input_image', type=str, help='Input image path for single mode')
    parser.add_argument('--output_image', type=str, help='Output image path for single mode')
    
    # Dataset mode
    parser.add_argument('--test_dataset', type=str, help='Test dataset type (UIEB or LSUI)')
    parser.add_argument('--data_root', type=str, help='Dataset root directory')
    
    # Batch mode
    parser.add_argument('--input_dir', type=str, help='Input directory for batch mode')
    
    # General options
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for batch mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    if args.device:
        config['device'] = args.device
    
    # Create tester
    tester = Tester(config, args.model_path)
    
    if args.mode == 'single':
        if not args.input_image:
            raise ValueError("Input image path is required for single mode")
        
        output_path = args.output_image or f"{Path(args.input_image).stem}_enhanced.png"
        enhanced, inference_time = tester.test_single_image(
            args.input_image, output_path, args.num_steps
        )
        print(f"Inference time: {inference_time:.3f}s")
    
    elif args.mode == 'dataset':
        if not args.test_dataset or not args.data_root:
            raise ValueError("Test dataset and data root are required for dataset mode")
        
        dataset_config = {
            'dataset_type': args.test_dataset,
            'data_root': args.data_root
        }
        
        metrics = tester.test_dataset(dataset_config, args.output_dir, args.num_steps)
    
    elif args.mode == 'batch':
        if not args.input_dir:
            raise ValueError("Input directory is required for batch mode")
        
        # Get all image files in input directory
        input_dir = Path(args.input_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(input_dir.glob(f'*{ext}')))
            image_paths.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        if not image_paths:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Found {len(image_paths)} images")
        tester.test_batch(image_paths, args.output_dir, args.batch_size, args.num_steps)
    
    elif args.mode == 'benchmark':
        results = tester.benchmark_speed()
        
        # Save benchmark results
        output_path = Path(args.output_dir) / 'benchmark_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()
