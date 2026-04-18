import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_isic_by_lesion(csv_path, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    严格按照 lesion_id 划分数据集，防止数据泄露
    """
    print(f"正在读取原始元数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. 提取病灶独立信息 (每个 lesion_id 对应一个确定的 dx 类别)
    # 这是关键：我们划分的是“人”，而不是“照片”
    lesion_df = df.groupby('lesion_id').agg({
        'dx': 'first'  # 假设同一个病灶的诊断是一致的
    }).reset_index()

    print(f"总图片数: {len(df)} | 独立病灶数: {len(lesion_df)}")

    # 2. 第一次划分：划出测试集 (Test)
    # stratify=lesion_df['dx'] 确保各子集类别比例与原图一致 (分层抽样)
    train_val_lesions, test_lesions = train_test_split(
        lesion_df,
        test_size=test_ratio,
        stratify=lesion_df['dx'],
        random_state=42
    )

    # 3. 第二次划分：从剩下的病灶中划出验证集 (Val)
    # 计算验证集在剩余部分中的比例
    val_adj_ratio = val_ratio / (train_ratio + val_ratio)
    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=val_adj_ratio,
        stratify=train_val_lesions['dx'],
        random_state=42
    )

    # 4. 根据划分好的 lesion_id，映射回所有的原始图片
    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])]
    val_df = df[df['lesion_id'].isin(val_lesions['lesion_id'])]
    test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])]

    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'isic_train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'isic_val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'isic_test_split.csv'), index=False)

    # 打印最终统计表，你可以直接复制到论文里
    print("\n" + "=" * 40)
    print("最终数据集划分")
    print("=" * 40)
    stats = []
    for name, d in zip(['Training', 'Validation', 'Testing'], [train_df, val_df, test_df]):
        print(f"{name:10} | 图片数: {len(d):5} | 占比: {len(d) / len(df) * 100:5.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    # 请修改为你的 HAM10000_metadata.csv 真实路径
    ORIGINAL_CSV = r"D:\Pycharm\ISIC_project\HAM10000\HAM10000_metadata.csv"
    OUTPUT_PATH = r"D:\Pycharm\ISIC_project\ISIC_Splits"

    split_isic_by_lesion(ORIGINAL_CSV, OUTPUT_PATH)