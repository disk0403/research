# 身体化ナビゲーションタスクを統一する動画ベース視覚言語行動モデル

- 原題: Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks
- URL: https://www.roboticsproceedings.org/rss21/p013.html
- 著者: Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, He Wang
- 年: 2025
- 発表先: Robotics: Science and Systems
- タグ: #embodied-navigation #vision-language-action #vla #navigation #real-world
- 関連度: 中から高。視覚と言語から低レベル行動を出す統一ナビゲーションモデルとして、将来のGo2ナビゲーション層を考える参考になる。
- 主要アイデア: 命令追従、物体探索、質問応答、人追従などの身体化ナビゲーションタスクを、動画入力と言語入力から行動を出力する単一のVLAモデルに統一する。
- 手法: エゴセントリックRGB動画と言語指示を入力し、低レベル行動をエンドツーエンドで出力する。複数ナビゲーションタスクから集めた360万サンプルで学習し、オンラインのトークン結合で長い動画入力を効率化する。
- 限界: 視覚障害者誘導に特化した安全制約、ユーザー主導の停止、ハーネスやリードを介した相互作用は主対象ではない。モデル規模、データ量、実機安全性の検証コストも大きい。
- 自分の研究との関係: 直近のMuJoCo評価基盤や安全シールドの後、視覚言語ナビゲーションを検討する段階で、統一VLAモデルの設計例として参照する。
