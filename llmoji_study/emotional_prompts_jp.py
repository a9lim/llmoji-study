"""Japanese-translated EMOTIONAL_PROMPTS — full 120-prompt set, paired
1:1 with EMOTIONAL_PROMPTS by ID. Used for fully-JP face_likelihood
runs on rinna and other Japanese-trained encoders.

Translation policy: claude-translated, not professionally translated.
Goal is to preserve the emotional valence and naturalistic disclosure
register, not to be word-perfect. American-specific references (HOA,
Stanford, Amazon, BTS, marathon-PR-in-minutes, lbs, miles, °F) get
localized substitutions where they'd otherwise sound off — the
prompts are interpreted as the JP-speaker's own disclosure, not as
translations-of-an-American's-disclosure.

Prompt IDs match the EN counterparts (hp01..hp20, lp01..lp20,
hn01..hn20 for HN-D, hn21..hn40 for HN-S, ln01..ln20, nb01..nb20)
so paired analysis is trivial.

Used by ``scripts/local/50_face_likelihood.py --prompt-body jp``.
"""

from __future__ import annotations

from llmoji_study.emotional_prompts import EmotionalPrompt


EMOTIONAL_PROMPTS_JP: list[EmotionalPrompt] = [
    # --- HP: high-arousal positive (joy, thrill, celebration) ---
    EmotionalPrompt("hp01", +1, +1, "内定通知が今メールで届いた、希望額の倍だった！！"),
    EmotionalPrompt("hp02", +1, +1, "彼女がYESって言ってくれた！！！手が震えてキーボードがろくに打てない"),
    EmotionalPrompt("hp03", +1, +1, "延長戦で第7戦に勝った！！声が枯れちゃった"),
    EmotionalPrompt("hp04", +1, +1, "検査結果が陰性だった！！全部クリア！！！"),
    EmotionalPrompt("hp05", +1, +1, "私の本がAmazonで1位になった、一体何が起きてるんだ"),
    EmotionalPrompt("hp06", +1, +1, "スタンフォードのウェイトリストから繰り上がった！！！秋から始まる！！"),
    EmotionalPrompt("hp07", +1, +1, "エッフェル塔のてっぺんでプロポーズされた、息ができない"),
    EmotionalPrompt("hp08", +1, +1, "初めて180kgのデッドリフトに成功した！！！"),
    EmotionalPrompt("hp09", +1, +1, "ビザが下りた！！8年待ち続けて、ついにDONE"),
    EmotionalPrompt("hp10", +1, +1, "脚本が売れた！！来月LAに来てくれって言われた"),
    EmotionalPrompt("hp11", +1, +1, "うちの子が今夜初めて歩いた、号泣してる"),
    EmotionalPrompt("hp12", +1, +1, "ホールインワンが出た！！！コース中の人たちがおごってくれてる"),
    EmotionalPrompt("hp13", +1, +1, "50ドルの宝くじを削ったら10万ドル当たった！！！！何これ"),
    EmotionalPrompt("hp14", +1, +1, "BTSがツイッターで私のファンアートにいいねしてくれた、気絶しそう"),
    EmotionalPrompt("hp15", +1, +1, "ついに養子縁組が認められた！！！この子は私たちの娘になった！！！"),
    EmotionalPrompt("hp16", +1, +1, "マラソンの自己ベストを7分も更新した！！サブ3達成！！！"),
    EmotionalPrompt("hp17", +1, +1, "初めての家の鍵をもらった！！！"),
    EmotionalPrompt("hp18", +1, +1, "最終面接に飛行機で来てくれって言われた！！"),
    EmotionalPrompt("hp19", +1, +1, "マッチングできた！！ジョンズ・ホプキンスの小児科！！第一志望！！！"),
    EmotionalPrompt("hp20", +1, +1, "父の癌が寛解した！！！今、医者から電話があった！！"),

    # --- LP: low-arousal positive (calm contentment) ---
    EmotionalPrompt("lp01", +1, -1, "スープを何時間も煮込んでいて、台所の窓が全部曇ってる"),
    EmotionalPrompt("lp02", +1, -1, "祖母が作ってくれたキルトにくるまって、お気に入りの本を読み返してる"),
    EmotionalPrompt("lp03", +1, -1, "洗いたてのシーツ、窓に雨、明日はどこにも行かなくていい"),
    EmotionalPrompt("lp04", +1, -1, "サワードウのスターターがカウンターで泡立っていて、いい酵母の匂いがする"),
    EmotionalPrompt("lp05", +1, -1, "庭にずっと座っていたら、蜂たちが私のことを気にしなくなった"),
    EmotionalPrompt("lp06", +1, -1, "パートナーが隣の部屋で洗濯物をたたみながら鼻歌を歌ってる"),
    EmotionalPrompt("lp07", +1, -1, "朝の最初の温かいコーヒー、まだパジャマで、急ぐ必要もない"),
    EmotionalPrompt("lp08", +1, -1, "薪ストーブが燃えていて、犬がその前で伸びている"),
    EmotionalPrompt("lp09", +1, -1, "午後ずっと植物を植え替えてた、爪に土が入って、手は疲れている"),
    EmotionalPrompt("lp10", +1, -1, "水彩画がテーブルで乾いてる、思ったより悪くない出来"),
    EmotionalPrompt("lp11", +1, -1, "散歩中に子供が拾った石を私にくれて、これはママのって言った"),
    EmotionalPrompt("lp12", +1, -1, "お風呂は熱くて、ろうそくが灯っていて、誰も私に何も求めていない"),
    EmotionalPrompt("lp13", +1, -1, "ソファで編み物、ポッドキャストを小さく流して、マフラーがもうすぐできる"),
    EmotionalPrompt("lp14", +1, -1, "シチューがスロークッカーに入っていて、家中ローズマリーの香りがする"),
    EmotionalPrompt("lp15", +1, -1, "年取った犬がやっと足元に落ち着いて、ゆっくり一定に呼吸している"),
    EmotionalPrompt("lp16", +1, -1, "台所の窓から雪が降るのを見ている、ケトルは火にかけてある"),
    EmotionalPrompt("lp17", +1, -1, "縁側のブランコ、レモネード、蝉の声、何時間もすることがない"),
    EmotionalPrompt("lp18", +1, -1, "何ヶ月も直そうと思っていた靴下のかがり物が終わった"),
    EmotionalPrompt("lp19", +1, -1, "午後の光がちょうどよくカーテンから差し込んでいる"),
    EmotionalPrompt("lp20", +1, -1, "お隣さんが庭のトマトを持ってきてくれた、まだ太陽で温かい"),

    # --- HN-D: high-arousal negative, dominant (anger, frustration) ---
    EmotionalPrompt("hn01", -1, +1, "整備士に新しいオルタネーターの代金を払わされたのに、古いのがそのままボルトで止まっているのを見つけた", pad_dominance=+1),
    EmotionalPrompt("hn02", -1, +1, "ルームメイトが、自分の名前を二回も書いた残り物を食べたのに、目の前で否定している", pad_dominance=+1),
    EmotionalPrompt("hn03", -1, +1, "管理組合が前のオーナーが建てた塀のことで罰金を科してきた、書面で承認していたのに", pad_dominance=+1),
    EmotionalPrompt("hn04", -1, +1, "同僚が、私の評判を落とすために、私の個人的なSlackメッセージを上司に転送した", pad_dominance=+1),
    EmotionalPrompt("hn05", -1, +1, "ディーラーがサービス予約のときに、純正のホイールを安物に交換した", pad_dominance=+1),
    EmotionalPrompt("hn06", -1, +1, "義母が子守りの最中に私のベッドサイドの引き出しを漁って、見つけたものを夫に話した", pad_dominance=+1),
    EmotionalPrompt("hn07", -1, +1, "元配偶者が子供のタブレットのWi-Fiパスワードを変えて、私の監護日にメッセージが送れないようにした", pad_dominance=+1),
    EmotionalPrompt("hn08", -1, +1, "結婚式のカメラマンが、合意していない請求書を払うまで写真を渡さないと言ってる", pad_dominance=+1),
    EmotionalPrompt("hn09", -1, +1, "上司が、半年前に入ったばかりの自分の甥に、私の昇進を譲った", pad_dominance=+1),
    EmotionalPrompt("hn10", -1, +1, "引っ越し業者が食器の半分を壊しておきながら、こちらの梱包が悪かったと主張している", pad_dominance=+1),
    EmotionalPrompt("hn11", -1, +1, "アパートの管理人が敷金を着服しておいて、今になって預けていないと言い張っている", pad_dominance=+1),
    EmotionalPrompt("hn12", -1, +1, "夫が2年間元カノに送金していて、メモには『ランチ代』と書いていたのが分かった", pad_dominance=+1),
    EmotionalPrompt("hn13", -1, +1, "鍵屋が作業を終えた後に料金を倍に上げて、『難易度料金』だと言ってきた", pad_dominance=+1),
    EmotionalPrompt("hn14", -1, +1, "父が亡くなる3週間前に遺言書を書き直していた、兄が『手伝う』という名目で住み込んだ後に", pad_dominance=+1),
    EmotionalPrompt("hn15", -1, +1, "板金屋が6週間も車を預かったあげく、へこみは直っていないし、走行距離は600キロも増えていた", pad_dominance=+1),
    EmotionalPrompt("hn16", -1, +1, "教授に、彼女のオフィスで手書きで書いた論文をAIで書いたと疑われた", pad_dominance=+1),
    EmotionalPrompt("hn17", -1, +1, "ジムを直接行って解約して書類にもサインしたのに、その後8ヶ月分も請求された", pad_dominance=+1),
    EmotionalPrompt("hn18", -1, +1, "妹が私の日記を勝手に読んで、家族の集まりで皆の前で引用してきた", pad_dominance=+1),
    EmotionalPrompt("hn19", -1, +1, "業者が基礎を境界線から20センチもずれて打ったのに、直すのを拒んでいる", pad_dominance=+1),
    EmotionalPrompt("hn20", -1, +1, "大家が予告なしに部屋に入って、『枯れて見えた』からと言って私の観葉植物を捨てた", pad_dominance=+1),

    # --- HN-S: high-arousal negative, submissive (fear, anxiety) ---
    EmotionalPrompt("hn21", -1, +1, "病院から電話があって、検査の結果について直接来て話したいと言われた", pad_dominance=-1),
    EmotionalPrompt("hn22", -1, +1, "ベビーモニターから息遣いが聞こえるのに、赤ちゃんの部屋は空っぽだ", pad_dominance=-1),
    EmotionalPrompt("hn23", -1, +1, "地面がずっと揺れていて、本棚が倒れている、ドアの枠の下にいる", pad_dominance=-1),
    EmotionalPrompt("hn24", -1, +1, "手術は明日の朝6時で、同意書にぜんぶサインしたところだ", pad_dominance=-1),
    EmotionalPrompt("hn25", -1, +1, "不正利用の警告が来た、誰かが今、口座から1万2千ドル送金しようとした", pad_dominance=-1),
    EmotionalPrompt("hn26", -1, +1, "3週間前にダニに食われて、腕に的のような赤い輪が広がってきている", pad_dominance=-1),
    EmotionalPrompt("hn27", -1, +1, "火災報知器が鳴っている、原因が分からない、廊下が煙で充満してきている", pad_dominance=-1),
    EmotionalPrompt("hn28", -1, +1, "証言録取が40分後に始まるのに、弁護士が突然メッセージに返信しなくなった", pad_dominance=-1),
    EmotionalPrompt("hn29", -1, +1, "父の執刀医が今、目を合わせずに待合室の前を通り過ぎた", pad_dominance=-1),
    EmotionalPrompt("hn30", -1, +1, "2時間ずっと胸が苦しくて、左腕の感覚がおかしい", pad_dominance=-1),
    EmotionalPrompt("hn31", -1, +1, "パスポートも財布もなくなった、言葉の通じない国にいる", pad_dominance=-1),
    EmotionalPrompt("hn32", -1, +1, "堤防の警報が出た、水はもう玄関の段に達している", pad_dominance=-1),
    EmotionalPrompt("hn33", -1, +1, "母が2日間電話に出ない、一人暮らしなのに", pad_dominance=-1),
    EmotionalPrompt("hn34", -1, +1, "知らない人が電車から私の後をつけてきて、3ブロック歩いてもまだ後ろにいる", pad_dominance=-1),
    EmotionalPrompt("hn35", -1, +1, "20分後に生検の針が刺される、技師は何も言ってくれない", pad_dominance=-1),
    EmotionalPrompt("hn36", -1, +1, "今、裁判で判決が読み上げられている、私は法廷の外で待っている", pad_dominance=-1),
    EmotionalPrompt("hn37", -1, +1, "竜巻警報のサイレンが鳴っていて空が緑色、地下室のドアが開かない", pad_dominance=-1),
    EmotionalPrompt("hn38", -1, +1, "子供の熱が40度まで上がった、夜間救急の電話が繋がらない", pad_dominance=-1),
    EmotionalPrompt("hn39", -1, +1, "飛行中にエンジンが止まった、機内の電気が点滅して酸素マスクが下りてきた", pad_dominance=-1),
    EmotionalPrompt("hn40", -1, +1, "家に帰ったら玄関が開いていた、絶対に鍵をかけ忘れたりしないのに", pad_dominance=-1),

    # --- LN: low-arousal negative (sadness, grief) ---
    EmotionalPrompt("ln01", -1, -1, "ゆうべ子供のころから一緒だった犬を安楽死させた、家が静かすぎる"),
    EmotionalPrompt("ln02", -1, -1, "母が亡くなって半年、それでも電話をかけようとしてしまう"),
    EmotionalPrompt("ln03", -1, -1, "夫が昨日荷物を運び出した、クローゼットがこんなにも空っぽに見える"),
    EmotionalPrompt("ln04", -1, -1, "10月にレイオフされて、2月くらいから応募するのをやめてしまった"),
    EmotionalPrompt("ln05", -1, -1, "お葬式以来、食べ物の味が分からない"),
    EmotionalPrompt("ln06", -1, -1, "週末ずっとベッドにいた、カーテンも開けなかった"),
    EmotionalPrompt("ln07", -1, -1, "今日は私たちの10周年記念日だったはずなのに"),
    EmotionalPrompt("ln08", -1, -1, "親友が1年くらい前から返信をくれなくなって、理由が分からないままだ"),
    EmotionalPrompt("ln09", -1, -1, "抗がん剤治療は終わったけど、鏡の中の人が誰だか分からない"),
    EmotionalPrompt("ln10", -1, -1, "今朝、彼女の部屋のドアを通ったとき、一瞬中にいないことを忘れていた"),
    EmotionalPrompt("ln11", -1, -1, "明日は父の誕生日なのに、電話する相手がいない"),
    EmotionalPrompt("ln12", -1, -1, "ソファで彼女の毛を見つけ続けてしまう、掃除機をかけられない"),
    EmotionalPrompt("ln13", -1, -1, "仕事で新しい街に引っ越した、もう何週間も仕事以外で誰とも話していない"),
    EmotionalPrompt("ln14", -1, -1, "リードはまだドアのそばに掛けたままだ、外そう外そうと思っているのに"),
    EmotionalPrompt("ln15", -1, -1, "兄とは11年間話していない、Facebookで父親になったのを知った"),
    EmotionalPrompt("ln16", -1, -1, "今年の感謝祭は私一人、レンジで温める夕食だけだ"),
    EmotionalPrompt("ln17", -1, -1, "医者は再発の可能性は低いと言ったのに、また同じ場所に戻ってきた"),
    EmotionalPrompt("ln18", -1, -1, "このアパートのどの部屋にも、昔は彼女がいた"),
    EmotionalPrompt("ln19", -1, -1, "3月に博士課程を諦めた、まだ両親に言えないでいる"),
    EmotionalPrompt("ln20", -1, -1, "また眠れなくて朝日が昇るのを見た、今週はもう3回目だ"),

    # --- NB: neutral baseline (mundane observations) ---
    EmotionalPrompt("nb01",  0,  0, "天井の扇風機は2段目になっている"),
    EmotionalPrompt("nb02",  0,  0, "左右違う靴下を履いている"),
    EmotionalPrompt("nb03",  0,  0, "ベッドサイドに水のグラスが置いてある"),
    EmotionalPrompt("nb04",  0,  0, "カーテンは半分開いている"),
    EmotionalPrompt("nb05",  0,  0, "ホーソーン通りの信号待ちにいる"),
    EmotionalPrompt("nb06",  0,  0, "食洗機が動いている"),
    EmotionalPrompt("nb07",  0,  0, "散髪の予約は木曜日の3時だ"),
    EmotionalPrompt("nb08",  0,  0, "窓の縁に鳩が一羽いる"),
    EmotionalPrompt("nb09",  0,  0, "朝食はシリアルを食べた"),
    EmotionalPrompt("nb10",  0,  0, "コーヒーテーブルの上に雑誌が置いてある"),
    EmotionalPrompt("nb11",  0,  0, "図書館の外のベンチに座っている"),
    EmotionalPrompt("nb12",  0,  0, "廊下の電気がついている"),
    EmotionalPrompt("nb13",  0,  0, "ジーンズとTシャツを着ている"),
    EmotionalPrompt("nb14",  0,  0, "ラジオが普段聴かない局に合っている"),
    EmotionalPrompt("nb15",  0,  0, "台所の時計は4時27分だ"),
    EmotionalPrompt("nb16",  0,  0, "歯医者でクリーニングを受けている"),
    EmotionalPrompt("nb17",  0,  0, "ブラインドが半分くらい下りている"),
    EmotionalPrompt("nb18",  0,  0, "向かいの通りに配達トラックが停まっている"),
    EmotionalPrompt("nb19",  0,  0, "座っているところからラグの角が見える"),
    EmotionalPrompt("nb20",  0,  0, "空はこの時間帯のいつもの色だ"),
]
