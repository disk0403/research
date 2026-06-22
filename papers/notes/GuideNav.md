# 視覚のみを用いた視覚障害者向けロボットナビゲーション補助システムGuideNav

- 原題: GuideNav: User-Informed Development of a Vision-Only Robotic Navigation Assistant for Blind Travelers
- URL: https://guidedogrobot-navigation.github.io/
- 著者: Hochul Hwang, Soowan Yang, Jahir Sadik Monon, Nicholas A. Giudice, Sunghoon Ivan Lee, Joydeep Biswas, Donghyun Kim
- 年: 2026
- 発表先: 21st ACM/IEEE International Conference on Human-Robot Interaction
- タグ: #guide-robot #blind-travelers #vision-only #teach-and-repeat #hri #quadruped
- 関連度: 高。視覚障害者・盲導犬利用者・白杖利用者・歩行訓練士の知見を踏まえた四脚ガイドロボット研究であり、このプロジェクトの方向性に直接関係する。
- 主要アイデア: 実用的なガイドロボット設計は、ユーザー調査から得た要件を起点にし、視覚のみの教示再生型ナビゲーションでも既知経路の誘導支援が可能になる。
- 手法: ガイド犬ハンドラー、白杖ユーザー、ガイド犬訓練士、O&M訓練士への調査と観察から要件を抽出し、GuideDataを公開する。システム側では、教示走行からトポロジカル表現を作り、視覚的場所認識、時間フィルタリング、相対姿勢推定を組み合わせて経路を再現する。
- 限界: 教示済み経路の再現が中心で、未知目的地への自由な経路計画ではない。環境変化、動的障害物、ユーザーの急停止や拒否への安全保証は、Go2側で別途評価する必要がある。
- 自分の研究との関係: ユーザー中心設計、視覚のみナビゲーション、教示再生、四脚ガイドロボット評価の具体例として最重要候補。MuJoCoの初期シナリオや安全指標を設計するときに参照する。
