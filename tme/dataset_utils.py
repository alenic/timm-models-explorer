

dataset_info = {
    "ImageNet-1k Validation": {
        "filename": "results-imagenet.csv",
        "description": "The standard 50,000 image ImageNet-1k validation set. Model selection during training utilizes this validation set, so it is not a true test set.",
        "source": "http://image-net.org/challenges/LSVRC/2012/index",
        "paper": {
            "title": "ImageNet Large Scale Visual Recognition Challenge",
            "url": "https://arxiv.org/abs/1409.0575"
            }
        },

    "ImageNet 'Real Labels'": {
        "filename": "results-imagenet-real.csv",
        "description": "The usual ImageNet-1k validation set with a fresh new set of labels intended to improve on mistakes in the original annotation process.",
        "source": "https://github.com/google-research/reassessed-imagenet",
        "paper": {
            "title": "Are we done with ImageNet?",
            "url": "https://arxiv.org/abs/2006.07159"
            }
        },

    "ImageNet-Rendition": {
        "filename": "results-imagenet-r.csv",
        "description": "Renditions of 200 ImageNet classes resulting in 30,000 images for testing robustness.",
        "source": "https://github.com/hendrycks/imagenet-r",
        "paper": {
            "title": "The Many Faces of Robustness",
            "url": "https://arxiv.org/abs/2006.16241"
            }
        },

    "ImageNet-Rendition (Clean)": {
        "filename": "results-imagenet-r-clean.csv",
        "description": "Renditions of 200 ImageNet classes resulting in 30,000 images for testing robustness. Cleaned validation with same 200 classes",
        "source": "https://github.com/hendrycks/imagenet-r",
        "paper": {
            "title": "The Many Faces of Robustness",
            "url": "https://arxiv.org/abs/2006.16241"
            }
        },


    "ImageNetV2 Matched Frequency": {
        "filename": "results-imagenetv2-matched-frequency.csv",
        "description": "An ImageNet test set of 10,000 images sampled from new images roughly 10 years after the original. Care was taken to replicate the original ImageNet curation/sampling process.",
        "source": "https://github.com/modestyachts/ImageNetV2",
        "paper": {
            "title": "Do ImageNet Classifiers Generalize to ImageNet?",
            "url": "https://arxiv.org/abs/1902.10811"
            }
        },

    "ImageNet-Sketch": {
        "filename": "results-sketch.csv",
        "description": "50,000 non photographic (or photos of such) images (sketches, doodles, mostly monochromatic) covering all 1000 ImageNet classes.",
        "source": "https://github.com/HaohanWang/ImageNet-Sketch",
        "paper": {
            "title": "Learning Robust Global Representations by Penalizing Local Predictive Power",
            "url": "https://arxiv.org/abs/1905.13549"
            }
        },

    "ImageNet-Adversarial": {
        "filename": "results-imagenet-a.csv",
        "description": "A collection of 7500 images covering 200 of the 1000 ImageNet classes. Images are naturally occurring adversarial examples that confuse typical ImageNet classifiers. This is a challenging dataset, your typical ResNet-50 will score 0% top-1.",
        "source": "https://github.com/hendrycks/natural-adv-examples",
        "paper": {
            "title": "Natural Adversarial Examples",
            "url": "https://arxiv.org/abs/1907.07174"
            }
        },

    "ImageNet-Adversarial (Clean)": {
        "filename": "results-imagenet-a.csv",
        "description": "A collection of 7500 images covering 200 of the 1000 ImageNet classes. Images are naturally occurring adversarial examples that confuse typical ImageNet classifiers. This is a challenging dataset, your typical ResNet-50 will score 0% top-1. Cleaned validation with same 200 classes",
        "source": "https://github.com/hendrycks/natural-adv-examples",
        "paper": {
            "title": "Natural Adversarial Examples",
            "url": "https://arxiv.org/abs/1907.07174"
            }
        },
}