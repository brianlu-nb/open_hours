from hours_inference import inference
from generate_hours import generate_data, current_path

if __name__ == '__main__':
    generate_data(1000, (0.8, 0.1, 0.1))
    inference(f'{current_path()}/train.jsonl')
    pass