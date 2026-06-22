# 身体化ナビゲーション基盤モデル

- 原題: Embodied Navigation Foundation Model
- URL: https://openreview.net/forum?id=kkBOIsrCXh
- 著者: Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, He Wang
- 年: 2026
- 発表先: International Conference on Learning Representations
- タグ: #embodied-navigation #foundation-model #vision-language-action #cross-embodiment #navigation
- 関連度: 中から高。四脚、ドローン、車輪型ロボット、車両などをまたぐナビゲーション基盤モデルであり、Go2を含む複数身体への一般化を考える材料になる。
- 主要アイデア: タスクと身体の違いをまたいで動作するナビゲーション基盤モデルNavFoMを作り、個別タスクごとの専用モデル依存を減らす。
- 手法: 複数カメラ構成と時間履歴を扱う統一アーキテクチャを用い、カメラ視点と時間文脈を示す識別トークンを導入する。800万件規模のナビゲーションサンプルで学習し、トークン長制約の下で履歴を動的にサンプリングする。
- 限界: 視覚障害者向け誘導、ユーザーとの物理的相互作用、明示的な安全停止は中心課題ではない。基盤モデルとして計算資源とデータ依存が大きく、実機Go2にそのまま載せるには軽量化や安全層が必要。
- 自分の研究との関係: 初期のウェイポイント追従や安全評価が安定した後、Go2の視覚言語ナビゲーションや複数タスク対応を検討する際の長期的な参考文献にする。
