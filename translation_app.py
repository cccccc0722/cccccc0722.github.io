import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
import nltk
import os

nltk.download('punkt', quiet=True)

LOCAL_MODEL_PATH = "./translation_model"

# 英汉字典 - 用于基于规则的逐词翻译
WORD_DICTIONARY = {
    "a": "一", "an": "一个", "the": "这", "this": "这个", "that": "那个",
    "is": "是", "are": "是", "was": "是", "were": "是", "be": "是",
    "i": "我", "you": "你", "he": "他", "she": "她", "it": "它",
    "we": "我们", "they": "他们", "my": "我的", "your": "你的",
    "his": "他的", "her": "她的", "its": "它的", "our": "我们的",
    "am": "是", "have": "有", "has": "有", "had": "有",
    "do": "做", "does": "做", "did": "做", "doing": "做",
    "go": "去", "goes": "去", "went": "去", "going": "去",
    "come": "来", "comes": "来", "came": "来", "coming": "来",
    "eat": "吃", "eats": "吃", "ate": "吃", "eating": "吃",
    "drink": "喝", "drinks": "喝", "drank": "喝", "drinking": "喝",
    "sleep": "睡", "sleeps": "睡", "slept": "睡", "sleeping": "睡",
    "work": "工作", "works": "工作", "worked": "工作", "working": "工作",
    "study": "学习", "studies": "学习", "studied": "学习", "studying": "学习",
    "read": "读", "reads": "读", "reading": "读",
    "write": "写", "writes": "写", "wrote": "写", "writing": "写",
    "say": "说", "says": "说", "said": "说", "saying": "说",
    "talk": "说话", "talks": "说话", "talked": "说话", "talking": "说话",
    "tell": "告诉", "tells": "告诉", "told": "告诉", "telling": "告诉",
    "think": "想", "thinks": "想", "thought": "想", "thinking": "想",
    "know": "知道", "knows": "知道", "knew": "知道", "knowing": "知道",
    "learn": "学习", "learns": "学习", "learned": "学习", "learning": "学习",
    "see": "看", "sees": "看", "saw": "看", "seeing": "看",
    "look": "看", "looks": "看", "looked": "看", "looking": "看",
    "watch": "观看", "watches": "观看", "watched": "观看", "watching": "观看",
    "listen": "听", "listens": "听", "listened": "听", "listening": "听",
    "hear": "听到", "hears": "听到", "heard": "听到", "hearing": "听到",
    "feel": "感觉", "feels": "感觉", "felt": "感觉", "feeling": "感觉",
    "good": "好的", "bad": "坏的", "happy": "高兴的", "sad": "伤心的",
    "big": "大的", "small": "小的", "old": "老的", "new": "新的",
    "hot": "热的", "cold": "冷的", "fast": "快的", "slow": "慢的",
    "red": "红的", "blue": "蓝的", "green": "绿的", "yellow": "黄的",
    "beautiful": "美丽的", "ugly": "丑的", "strong": "强壮的", "weak": "虚弱的",
    "yes": "是", "no": "否", "hello": "你好", "hi": "嗨",
    "goodbye": "再见", "bye": "再见", "thank": "谢谢", "thanks": "谢谢",
    "please": "请", "sorry": "对不起", "excuse": "原谅",
    "morning": "早晨", "afternoon": "下午", "evening": "晚上", "night": "夜晚",
    "today": "今天", "tomorrow": "明天", "yesterday": "昨天",
    "day": "天", "week": "周", "month": "月", "year": "年",
    "time": "时间", "clock": "时钟", "hour": "小时", "minute": "分钟",
    "water": "水", "food": "食物", "eat": "吃", "drink": "喝",
    "book": "书", "pen": "笔", "paper": "纸", "school": "学校",
    "home": "家", "house": "房子", "car": "汽车", "bus": "公交车",
    "apple": "苹果", "orange": "橙子", "banana": "香蕉", "fruit": "水果",
    "cat": "猫", "dog": "狗", "bird": "鸟", "fish": "鱼",
    "love": "爱", "like": "喜欢", "hate": "讨厌", "want": "想要",
    "need": "需要", "help": "帮助", "use": "使用", "make": "制作",
    "find": "找到", "get": "得到", "give": "给予", "take": "拿",
    "put": "放", "keep": "保持", "let": "让", "start": "开始",
    "stop": "停止", "continue": "继续", "finish": "完成", "end": "结束",
    "now": "现在", "then": "然后", "here": "这里", "there": "那里",
    "very": "非常", "so": "所以", "but": "但是", "and": "和",
    "or": "或", "if": "如果", "because": "因为", "why": "为什么",
    "what": "什么", "when": "什么时候", "where": "哪里", "who": "谁",
    "how": "如何", "many": "许多", "much": "许多", "some": "一些",
    "any": "任何", "all": "全部", "no": "没有", "not": "不",
    "will": "将", "would": "将", "should": "应该", "can": "能够",
    "could": "能够", "may": "可能", "might": "可能", "must": "必须"
}

SLANG_DICTIONARY = {
    "burning the midnight oil": {"translation": "熬夜工作", "explanation": "指工作到深夜"},
    "crunch time": {"translation": "关键时刻", "explanation": "指需要全力以赴的紧要关头"},
    "swim or sink": {"translation": "成败在此一举", "explanation": "指要么成功要么失败"},
    "cutting corners": {"translation": "偷工减料", "explanation": "指为了省事而牺牲质量"},
    "go the extra mile": {"translation": "加倍努力", "explanation": "指付出额外的努力"},
    "think outside the box": {"translation": "跳出思维定式", "explanation": "指用创新方式思考"},
    "hit the ground running": {"translation": "立即投入工作", "explanation": "指迅速开始并高效运作"},
    "break a leg": {"translation": "祝你好运", "explanation": "戏剧界的祝福语"},
    "piece of cake": {"translation": "小菜一碟", "explanation": "指非常容易的事情"},
    "cost an arm and a leg": {"translation": "价格昂贵", "explanation": "指某物非常贵"},
    "cat's out of the bag": {"translation": "秘密泄露", "explanation": "指秘密被公开"},
    "spill the beans": {"translation": "泄露秘密", "explanation": "指无意中说出秘密"},
    "under the weather": {"translation": "身体不适", "explanation": "指感觉不舒服"},
    "over the moon": {"translation": "欣喜若狂", "explanation": "指非常高兴"},
    "down in the dumps": {"translation": "情绪低落", "explanation": "指心情沮丧"},
    "head over heels": {"translation": "深深爱上", "explanation": "指坠入爱河"},
    "heart of gold": {"translation": "心地善良", "explanation": "指人非常善良"},
    "bite the bullet": {"translation": "硬着头皮", "explanation": "指勇敢面对困难"},
    "fish out of water": {"translation": "格格不入", "explanation": "指在陌生环境中不自在"},
    "get cold feet": {"translation": "临阵退缩", "explanation": "指在最后时刻害怕而放弃"},
    "go with the flow": {"translation": "随波逐流", "explanation": "指顺其自然"},
    "jump on the bandwagon": {"translation": "跟风", "explanation": "指跟随潮流"},
    "keep your chin up": {"translation": "振作起来", "explanation": "指保持乐观"},
    "leave no stone unturned": {"translation": "千方百计", "explanation": "指想尽一切办法"},
    "not rocket science": {"translation": "小菜一碟", "explanation": "指事情并不复杂"},
    "on cloud nine": {"translation": "心花怒放", "explanation": "指非常快乐"},
    "play it by ear": {"translation": "随机应变", "explanation": "指不按计划，根据情况行事"},
    "pull someone's leg": {"translation": "开玩笑", "explanation": "指戏弄某人"},
    "turn a blind eye": {"translation": "视而不见", "explanation": "指故意忽视"},
    "when pigs fly": {"translation": "绝不可能", "explanation": "指某件事永远不会发生"},
    "throw caution to the wind": {"translation": "不顾一切", "explanation": "指冒险行事"},
    "take with a grain of salt": {"translation": "半信半疑", "explanation": "指不完全相信"},
    "the ball is in your court": {"translation": "轮到你了", "explanation": "指责任在对方"},
    "break the ice": {"translation": "打破僵局", "explanation": "指开始对话或缓解紧张气氛"},
    "ring a bell": {"translation": "听起来耳熟", "explanation": "指感觉以前听过"},
    "miss the boat": {"translation": "错失良机", "explanation": "指错过机会"},
    "in the same boat": {"translation": "同舟共济", "explanation": "指处境相同"},
    "pass the buck": {"translation": "推卸责任", "explanation": "指把责任推给别人"},
    "call it a day": {"translation": "收工", "explanation": "指结束一天的工作"},
    "the early bird catches the worm": {"translation": "早起的鸟儿有虫吃", "explanation": "指勤奋有回报"},
    "raining cats and dogs": {"translation": "倾盆大雨", "explanation": "指下很大的雨"},
    "every cloud has a silver lining": {"translation": "黑暗中总有一线光明", "explanation": "指坏事中总有好的一面"},
    "don't count your chickens before they hatch": {"translation": "不要高兴太早", "explanation": "指事情未完成前不要过早乐观"},
    "actions speak louder than words": {"translation": "行动胜于言语", "explanation": "指实际行动比空谈更重要"},
    "a picture is worth a thousand words": {"translation": "一图胜千言", "explanation": "指图像比文字更有表现力"},
    "kill two birds with one stone": {"translation": "一石二鸟", "explanation": "指一举两得"},
    "no pain, no gain": {"translation": "一分耕耘，一分收获", "explanation": "指没有付出就没有回报"},
    "practice makes perfect": {"translation": "熟能生巧", "explanation": "指练习能提高技能"},
    "the grass is always greener on the other side": {"translation": "这山望着那山高", "explanation": "指总觉得别人的处境更好"},
    "when in Rome, do as the Romans do": {"translation": "入乡随俗", "explanation": "指在不同地方要遵从当地习俗"},
    "time flies": {"translation": "时光飞逝", "explanation": "指时间过得很快"},
    "money talks": {"translation": "金钱万能", "explanation": "指金钱有很大影响力"},
    "blood is thicker than water": {"translation": "血浓于水", "explanation": "指亲情比其他关系更重要"},
    "better late than never": {"translation": "迟到总比不到好", "explanation": "指做了总比不做好"},
    "barking up the wrong tree": {"translation": "找错对象", "explanation": "指找错目标或方向"},
    "don't put all your eggs in one basket": {"translation": "不要把鸡蛋放在一个篮子里", "explanation": "指不要孤注一掷"},
    "you can't judge a book by its cover": {"translation": "人不可貌相", "explanation": "指不能凭外表判断事物"},
    "all roads lead to Rome": {"translation": "条条大路通罗马", "explanation": "指达到目的有多种方法"},
    "slow and steady wins the race": {"translation": "稳扎稳打", "explanation": "指耐心和坚持能成功"},
    "birds of a feather flock together": {"translation": "物以类聚", "explanation": "指相似的人会聚集在一起"},
    "curiosity killed the cat": {"translation": "好奇心害死猫", "explanation": "指过于好奇可能带来麻烦"},
    "a stitch in time saves nine": {"translation": "小洞不补，大洞吃苦", "explanation": "指及时处理问题能避免更大麻烦"},
    "when the going gets tough, the tough get going": {"translation": "越挫越勇", "explanation": "指困难时勇敢面对"},
    "it's not over till it's over": {"translation": "不到最后不言放弃", "explanation": "指事情未结束前仍有希望"},
    "the show must go on": {"translation": "演出必须继续", "explanation": "指不管发生什么都要坚持"},
    "to each his own": {"translation": "各有所好", "explanation": "指每个人有不同的喜好"},
    "the apple doesn't fall far from the tree": {"translation": "有其父必有其子", "explanation": "指子女与父母相似"},
    "spinning my wheels": {"translation": "原地打转", "explanation": "指没有进展"},
    "full-blown": {"translation": "全面爆发的", "explanation": "指完全发展的状态"},
    "a tall order": {"translation": "艰巨任务", "explanation": "指很难完成的要求"},
    "thin ice": {"translation": "如履薄冰", "explanation": "指处于危险境地"},
    "burning issue": {"translation": "热点问题", "explanation": "指急需解决的问题"},
    "tip of the iceberg": {"translation": "冰山一角", "explanation": "指只是问题的一小部分"},
    "playing hardball": {"translation": "强硬手段", "explanation": "指采取强硬态度"},
    "win hands down": {"translation": "轻松获胜", "explanation": "指毫无悬念地赢"},
    "last-ditch effort": {"translation": "最后努力", "explanation": "指最后的尝试"},
    "back against the wall": {"translation": "走投无路", "explanation": "指处于绝境"},
    "by the skin of your teeth": {"translation": "侥幸逃脱", "explanation": "指勉强成功"},
    "in the driver's seat": {"translation": "掌控局面", "explanation": "指处于主导地位"},
    "out of the frying pan and into the fire": {"translation": "刚出狼窝又入虎口", "explanation": "指情况更糟"},
    "the writing is on the wall": {"translation": "大势已去", "explanation": "指明显的迹象"},
    "water under the bridge": {"translation": "既往不咎", "explanation": "指过去的事不再计较"},
    "white elephant": {"translation": "累赘", "explanation": "指昂贵而无用的东西"},
    "wild goose chase": {"translation": "徒劳无功", "explanation": "指白费力气"},
    "wolf in sheep's clothing": {"translation": "披着羊皮的狼", "explanation": "指伪装善良的坏人"},
    "world of difference": {"translation": "天壤之别", "explanation": "指巨大差异"},
    "worst-case scenario": {"translation": "最坏情况", "explanation": "指最糟糕的可能"},
    "yes man": {"translation": "应声虫", "explanation": "指唯唯诺诺的人"},
    "dark horse": {"translation": "黑马", "explanation": "指出人意料的获胜者"},
    "early adopter": {"translation": "早期使用者", "explanation": "指最先尝试新产品的人"},
    "game plan": {"translation": "策略", "explanation": "指行动计划"},
    "power play": {"translation": "权力游戏", "explanation": "指争权夺利"},
    "reality check": {"translation": "现实检验", "explanation": "指认清事实"},
    "rule of thumb": {"translation": "经验法则", "explanation": "指大致的准则"},
    "status quo": {"translation": "现状", "explanation": "指当前状态"},
    "target audience": {"translation": "目标受众", "explanation": "指针对的人群"},
    "brainstorming": {"translation": "头脑风暴", "explanation": "指集思广益"},
    "due diligence": {"translation": "尽职调查", "explanation": "指仔细审查"},
    "key performance indicators": {"translation": "关键绩效指标", "explanation": "指衡量成功的标准"},
    "low-hanging fruit": {"translation": "唾手可得", "explanation": "指容易实现的目标"},
    "paradigm shift": {"translation": "范式转变", "explanation": "指根本性改变"},
    "proactive": {"translation": "主动的", "explanation": "指提前行动"},
    "synergy": {"translation": "协同效应", "explanation": "指合作产生的额外效果"},
    "value proposition": {"translation": "价值主张", "explanation": "指提供的价值"},
    "big data": {"translation": "大数据", "explanation": "指海量数据"},
    "cloud computing": {"translation": "云计算", "explanation": "指远程计算服务"},
    "deep dive": {"translation": "深入分析", "explanation": "指详细研究"},
    "digital transformation": {"translation": "数字化转型", "explanation": "指技术革新"},
    "disruptive technology": {"translation": "颠覆性技术", "explanation": "指彻底改变行业的技术"},
    "fintech": {"translation": "金融科技", "explanation": "指金融与科技结合"},
    "machine learning": {"translation": "机器学习", "explanation": "指AI学习能力"},
    "network effect": {"translation": "网络效应", "explanation": "指用户越多价值越高"},
    "scalability": {"translation": "可扩展性", "explanation": "指增长能力"},
    "user experience": {"translation": "用户体验", "explanation": "指用户感受"},
    "virtual reality": {"translation": "虚拟现实", "explanation": "指模拟环境"},
    "artificial intelligence": {"translation": "人工智能", "explanation": "指智能技术"},
    "augmented reality": {"translation": "增强现实", "explanation": "指现实叠加虚拟"},
    "blockchain": {"translation": "区块链", "explanation": "指分布式账本"},
    "content marketing": {"translation": "内容营销", "explanation": "指通过内容吸引客户"},
    "data analytics": {"translation": "数据分析", "explanation": "指数据解读"},
    "freemium": {"translation": "免费增值", "explanation": "指免费基础+付费升级"},
    "growth hacking": {"translation": "增长黑客", "explanation": "指低成本快速增长"},
    "influencer marketing": {"translation": "网红营销", "explanation": "指通过意见领袖推广"},
    "product market fit": {"translation": "产品市场匹配", "explanation": "指产品满足市场需求"},
    "retention rate": {"translation": "留存率", "explanation": "指用户保留比例"},
    "search engine optimization": {"translation": "搜索引擎优化", "explanation": "指提升搜索排名"},
    "social media marketing": {"translation": "社交媒体营销", "explanation": "指社交平台推广"},
    "viral marketing": {"translation": "病毒式营销", "explanation": "指快速传播"},
    "agile": {"translation": "敏捷", "explanation": "指灵活快速的方法"},
    "sprint": {"translation": "冲刺", "explanation": "指短期高强度工作"},
    "standup": {"translation": "站会", "explanation": "指每日例会"},
    "user story": {"translation": "用户故事", "explanation": "指用户需求描述"},
    "bug fix": {"translation": "修复漏洞", "explanation": "指解决问题"},
    "code review": {"translation": "代码审查", "explanation": "指检查代码质量"},
    "debugging": {"translation": "调试", "explanation": "指查找错误"},
    "deployment": {"translation": "部署", "explanation": "指上线发布"},
    "integration testing": {"translation": "集成测试", "explanation": "指测试组件协作"},
    "regression testing": {"translation": "回归测试", "explanation": "指确保修改不破坏现有功能"},
    "unit testing": {"translation": "单元测试", "explanation": "指测试单个组件"},
}

def detect_and_replace_slang(text):
    detected_slang = []
    result_text = text
    
    for slang, info in SLANG_DICTIONARY.items():
        if slang.lower() in text.lower():
            detected_slang.append({
                "original": slang,
                "translation": info["translation"],
                "explanation": info["explanation"]
            })
            result_text = result_text.lower().replace(slang.lower(), info["translation"])
    
    return result_text, detected_slang

@st.cache_resource
def load_translator():
    if os.path.exists(LOCAL_MODEL_PATH) and len(os.listdir(LOCAL_MODEL_PATH)) > 0:
        try:
            tokenizer = MarianTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            model = MarianMTModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            return (model, tokenizer), "local"
        except Exception as e:
            return None, f"local_error: {str(e)}"
    
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
        return translator, "huggingface"
    except Exception as e:
        return None, f"remote_error: {str(e)}"

def translate_with_local_model(model_tuple, text):
    model, tokenizer = model_tuple
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 基于规则的逐词翻译函数
def rule_based_translation(text):
    # 清理标点符号
    cleaned_text = text.lower()
    # 基于空格分词
    words = cleaned_text.split()
    translated_words = []
    
    for word in words:
        # 去掉标点符号
        cleaned_word = word.strip(',.!?;:\"\'()[]{}')
        # 查找词典
        if cleaned_word in WORD_DICTIONARY:
            translated_words.append(WORD_DICTIONARY[cleaned_word])
        else:
            # 词典中没有的词保留英文
            translated_words.append(word)
    
    return ''.join(f"{w} " for w in translated_words).strip()

st.set_page_config(
    page_title="多模式翻译应用",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 多模式翻译应用")

tab1, tab2, tab3 = st.tabs(["神经机器翻译", "文本分析翻译", "双语对照翻译"])

with tab1:
    st.header("神经机器翻译 (英译中)")
    st.markdown("使用 Hugging Face Transformers 模型进行高质量翻译")
    
    use_slang_correction = st.checkbox("启用俚语优化", value=True, help="自动检测并优化英文俚语和习语的翻译")

    input_text = st.text_area(
        "输入英文句子:",
        placeholder="请输入要翻译的英文文本...",
        height=150
    )

    translate_button = st.button("翻译", type="primary")

    if translate_button and input_text.strip():
        detected_slang_info = []
        processed_text = input_text
        
        if use_slang_correction:
            processed_text, detected_slang_info = detect_and_replace_slang(input_text)

        with st.spinner("正在加载翻译模型..."):
            translator, status = load_translator()

        if translator is not None:
            with st.spinner("翻译中，请稍候..."):
                try:
                    if status == "local":
                        translation = translate_with_local_model(translator, processed_text)
                        st.success("✅ 翻译完成！(本地模型)")
                    else:
                        result = translator(processed_text)[0]
                        translation = result['translation_text']
                        st.success("✅ 翻译完成！(Hugging Face)")
                    
                    st.markdown("### 中文译文:")
                    st.info(translation)
                    
                    if detected_slang_info:
                        st.markdown("### 🔍 检测到的俚语/习语:")
                        for item in detected_slang_info:
                            with st.expander(f"「{item['original']}」→「{item['translation']}」"):
                                st.markdown(f"**解释**: {item['explanation']}")
                                st.markdown(f"**替换后**: 原文中的 `{item['original']}` 被替换为 `{item['translation']}`")
                except Exception as e:
                    st.error(f"翻译失败: {str(e)}")
        else:
            st.error(f"无法加载翻译模型: {status}")
            st.warning("请先运行 `python download_model.py` 下载模型")
            
            with st.expander("📥 手动下载模型步骤"):
                st.markdown("""
                **步骤1：运行下载脚本**
                ```bash
                python download_model.py
                ```

                **步骤2：如果脚本失败，请手动下载**
                1. 访问 https://hf-mirror.com/Helsinki-NLP/opus-mt-en-zh
                2. 下载所有文件
                3. 创建 `translation_model` 文件夹
                4. 将所有文件放入该文件夹

                **步骤3：重新运行应用**
                ```bash
                streamlit run translation_app.py
                ```
                """)
    elif translate_button:
        st.warning("请输入要翻译的文本!")

    if os.path.exists(LOCAL_MODEL_PATH) and len(os.listdir(LOCAL_MODEL_PATH)) > 0:
        st.success(f"✅ 检测到本地模型: {LOCAL_MODEL_PATH}")
    else:
        st.info("💡 提示：本地模型未找到，尝试从远程加载...")

with tab2:
    st.header("🔄 翻译效果对比：NMT vs 逐词直译")
    st.markdown("对比神经机器翻译与早期基于规则的逐词翻译的效果")
    
    input_text_compare = st.text_area(
        "输入英文句子进行对比:",
        placeholder="请输入要对比翻译的英文文本...",
        height=150
    )

    compare_button = st.button("开始对比", type="primary")

    if compare_button and input_text_compare.strip():
        with st.spinner("正在进行两种翻译..."):
            # 1. 基于规则的逐词翻译
            rule_based_result = rule_based_translation(input_text_compare)
            
            # 2. 神经机器翻译
            nmt_result = ""
            translator, status = load_translator()
            if translator is not None:
                try:
                    if status == "local":
                        nmt_result = translate_with_local_model(translator, input_text_compare)
                    else:
                        result = translator(input_text_compare)[0]
                        nmt_result = result['translation_text']
                except Exception as e:
                    nmt_result = f"翻译出错: {str(e)}"
            else:
                nmt_result = "无法加载翻译模型"

        # 并排展示两种结果
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 早期翻译：逐词直译")
            st.markdown("**（基于规则的机器翻译）**")
            st.warning(rule_based_result)
            st.markdown("""
            **特点：**
            - 基于预定义字典
            - 逐词翻译，不考虑语境
            - 语法错误较多
            - 不理解惯用语和俚语
            """)
        
        with col2:
            st.markdown("### 🤖 现代翻译：神经机器翻译")
            st.markdown("**（NMT - Neural Machine Translation）**")
            st.success(nmt_result)
            st.markdown("""
            **特点：**
            - 基于深度学习模型
            - 理解上下文和语义
            - 自然流畅的表达
            - 支持俚语和惯用语
            """)

        # 添加对比总结
        st.markdown("---")
        st.markdown("### 💡 对比总结")
        st.info("""
        通过对比可以清楚看到神经机器翻译的巨大优势：
        1. **语义理解**：NMT能理解句子的整体含义，而不是简单的单词拼接
        2. **语法正确**：NMT输出的中文符合汉语语法规则
        3. **流畅自然**：NMT的译文更符合人类语言表达习惯
        4. **语域适配**：NMT能根据语境选择合适的表达方式
        """)
    elif compare_button:
        st.warning("请输入要翻译的文本!")

with tab3:
    st.header("📊 机器翻译自动评测 (BLEU评分)")
    st.markdown("使用 BLEU 评分自动评估机器译文与参考译文的相似度")
    
    st.markdown("### 输入区域")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_text = st.text_area(
            "1. 英文原文 (Source)",
            placeholder="请输入要翻译的英文原文...",
            height=100
        )
    
    with col2:
        reference_text = st.text_area(
            "2. 参考译文 (Reference)",
            placeholder="请输入标准的中文参考译文...",
            height=100
        )
    
    with col3:
        candidate_text = st.text_area(
            "3. 候选译文 (Candidate)",
            placeholder="请输入机器生成的候选译文...",
            height=100
        )
    
    auto_generate_button = st.button("📝 自动生成候选译文", help="使用神经机器翻译自动生成候选译文")
    evaluate_button = st.button("🔍 开始评测", type="primary")
    
    if auto_generate_button and source_text.strip():
        with st.spinner("正在生成候选译文..."):
            translator, status = load_translator()
            if translator is not None:
                try:
                    if status == "local":
                        generated = translate_with_local_model(translator, source_text)
                    else:
                        result = translator(source_text)[0]
                        generated = result['translation_text']
                    st.session_state.candidate_text = generated
                    st.success(f"候选译文已生成！")
                except Exception as e:
                    st.error(f"生成失败: {str(e)}")
            else:
                st.error("无法加载翻译模型")
    
    if 'candidate_text' in st.session_state:
        candidate_text = st.session_state.candidate_text
    
    if evaluate_button:
        if not source_text.strip():
            st.warning("请输入英文原文!")
        elif not reference_text.strip():
            st.warning("请输入参考译文!")
        elif not candidate_text.strip():
            st.warning("请输入候选译文!")
        else:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            def tokenize_chinese(text):
                tokens = []
                for char in text:
                    if char.strip():
                        tokens.append(char)
                return tokens
            
            reference = [tokenize_chinese(reference_text)]
            candidate = tokenize_chinese(candidate_text)
            
            smoothing = SmoothingFunction()
            bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing.method4)
            bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing.method4)
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing.method4)
            bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method4)
            
            st.markdown("---")
            st.markdown("### 📈 BLEU 评分结果")
            
            col_scores = st.columns(4)
            with col_scores[0]:
                st.metric("BLEU-1 (单字匹配)", f"{bleu1:.4f}")
            with col_scores[1]:
                st.metric("BLEU-2 (双字匹配)", f"{bleu2:.4f}")
            with col_scores[2]:
                st.metric("BLEU-3 (三字匹配)", f"{bleu3:.4f}")
            with col_scores[3]:
                st.metric("BLEU-4 (四字匹配)", f"{bleu4:.4f}")
            
            st.markdown("### 🎯 综合评分")
            st.success(f"**BLEU-4 综合得分**: {bleu4:.4f}")
            
            if bleu4 >= 0.8:
                st.success("🎉 优秀！译文质量非常高")
            elif bleu4 >= 0.6:
                st.info("👍 良好！译文质量较好")
            elif bleu4 >= 0.4:
                st.warning("⚡ 一般！译文需要改进")
            else:
                st.error("❌ 较差！译文质量较低")
            
            st.markdown("---")
            st.markdown("### 📖 BLEU 分数说明")
            st.info("""
            **BLEU (Bilingual Evaluation Understudy)** 是机器翻译领域最常用的自动评测指标：
            
            | 分数范围 | 评价 |
            |----------|------|
            | 0.8 - 1.0 | 优秀：译文与参考译文高度一致 |
            | 0.6 - 0.8 | 良好：译文质量较好 |
            | 0.4 - 0.6 | 一般：译文基本准确但不够流畅 |
            | 0.0 - 0.4 | 较差：译文质量较低 |
            
            **不同N-gram的含义：**
            - **BLEU-1**：衡量单字匹配程度，反映词汇准确性
            - **BLEU-2**：衡量双字短语匹配，反映短语连贯性
            - **BLEU-3**：衡量三字短语匹配，反映句子结构
            - **BLEU-4**：衡量四字短语匹配，反映整体流畅度
            
            **注意**：BLEU分数是基于n-gram匹配的自动评估，虽然广泛使用，但无法完全替代人工评估，因为它无法完全理解语义和语境。
            """)
