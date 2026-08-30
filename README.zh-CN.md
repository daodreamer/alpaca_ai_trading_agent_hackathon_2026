*语言：* [English](README.md) · **简体中文**

# AlphaGate

**可以被否决的交易 agent。**

为 [Alpaca AI Trading Agents Hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)
而作（2026 年 8 月 28 日 – 9 月 4 日），Options Alpha Agents 赛道。

> **从来没接触过这类东西？** 直接跳到 [**从这里开始**](#从这里开始)——
> 那一节假设你什么都不懂，大约半小时后你会看到一个跑起来的 dashboard。
>
> **想知道它是怎么运作的？** 读
> [**架构与运作流程**](docs/ARCHITECTURE.zh-CN.md)——从一个猜想到一笔下出去的
> 订单，每个环节都有图。

---

## 从这里开始

*这一节假设你从来没做过这些事。如果你已经知道什么叫 paper trading 的 API key，
直接跳到 [使用手册](#使用手册)。*

### 这是什么？

这是一个替你自动买卖股票的程序。

它不寻常的地方在于**做事的顺序**。大多数这类程序从"猜什么会涨"开始。
这个程序从**检验**一个猜想开始：先拿十五年的历史数据检验，再拿两年它被刻意
禁止看过的数据检验一次。只有两关都活下来的猜想才被允许碰这个账户——
而且即使这样，它想下的每一单还得先过一段叫 **Risk Gate** 的代码，
那段代码唯一的工作就是说**不**。

可以把它想成同一栋楼里的一个科学家和一个安全检查员。科学家提议，检查员否决。

### 这是真钱吗？

**不是。** 它跑在 Alpaca 的 **paper trading（模拟盘）**账户上——一个免费的
练习账户，里面是假钱。行情是真的，钱是假的。

程序在设计上**无法**连到真钱账户，而且这一点被检查了两次：Alpaca 的真实 key
以 `AK` 开头、模拟盘 key 以 `PK` 开头；真实交易的网址和模拟盘的网址也不一样。
两个信号里只要有一个说"这是真的"，程序就拒绝启动。

**不要去改它。** 这个仓库里的任何东西都不构成投资建议，也没有任何人用真金白银
测试过它。

### 你需要什么

| | |
| --- | --- |
| 一台电脑 | Windows、macOS、Linux 都可以 |
| 能上网 | 它要和 Alpaca 的服务器说话 |
| 大约 30 分钟 | 大部分时间在等下载 |
| 一个免费的 Alpaca 账户 | 两分钟就能注册，不用充钱 |

你**不需要**会编程。你要做的是在终端里敲命令——终端就是一个你用打字而不是
点鼠标来下达指令的窗口。

- **Windows：** 按开始键，输入 `powershell`，回车。
- **macOS：** 按 ⌘ + 空格，输入 `terminal`，回车。
- **Linux：** 你知道它在哪。

### 第 1 步 — 装一个用来跑程序的工具

这个项目是用 Python 写的。但你不用自己装 Python，你只需要装一个叫 **uv**
的小程序，它会替你把合适版本的 Python 和所有依赖库都取回来。

把下面这一行复制到终端里，回车。

**Windows：**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS 或 Linux：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后**关掉终端，重新开一个**——这一步很重要，因为只有新开的窗口才知道
`uv` 存在了。验证一下：

```bash
uv --version
```

看到版本号就说明这一步完成了。

### 第 2 步 — 把代码拿下来

```bash
git clone <仓库地址>
cd hackathon_alpaca_ai_trading_20260828
```

如果你没有 `git`，就从仓库网页上下载 ZIP 压缩包，解压，然后用终端进入解压出来的
文件夹。

**这份指南里的每一条命令都是在那个文件夹里敲的。** 如果某条命令报"找不到"，
先确认你在对的位置——敲 `ls`（macOS/Linux）或 `dir`（Windows），
列表里应该能看到 `README.md`。

### 第 3 步 — 拿到你的免费模拟盘 key

1. 打开 [alpaca.markets](https://alpaca.markets) 注册。免费。
2. 登录后，找到 **Paper Trading** 那个开关，确保它是**打开**的。
   这就是模拟盘。
3. 找到 **API Keys**，点 **Generate New Key**。
4. 页面会给你两串很长的字母数字：一个 **Key ID**，一个 **Secret Key**。

**现在就把它们复制到别处保存。** Secret 那一串只会显示一次，之后再也看不到。
弄丢了就重新生成一对，没有任何损失。

确认 Key ID 是以 **`PK`** 开头的。如果是 `AK` 开头，那你拿到的是真钱账户的
key，说明 Paper Trading 开关没打开。回去打开它。

### 第 4 步 — 把 key 告诉程序

项目文件夹里有一个 `.env.example`。复制一份并命名为 `.env.local`：

**Windows：**

```powershell
copy .env.example .env.local
```

**macOS 或 Linux：**

```bash
cp .env.example .env.local
```

然后用任意文本编辑器（记事本就行）打开 `.env.local`，填三行：

```
ALPACA_API_KEY_ID=PK................
ALPACA_API_SECRET_KEY=................
ALPHAGATE_STRATEGY_FINGERPRINT=3f6e2c8a9309068b
```

前两个是第 3 步拿到的 key。第三个指明了**这个账户被允许交易哪一个策略**——
那串长字符是唯一一个通过了全部检验的策略的标识。程序拒绝执行其他任何东西，
而这个拒绝正是关键：没有它，往某个文件夹里丢一个不同的文件就会悄悄改变
你账户里持有的东西。

`.env.local` 不会被上传到任何地方，也被版本控制排除在外，所以你的 key
只留在你自己的电脑上。

### 第 5 步 — 检查一切是否正常

```bash
uv run --directory backend python -m alphagate equity-preflight
```

第一次运行时它会下载 Python 和一堆依赖库，这是最慢的一步。之后每次只要几秒。

你应该看到这样一份清单：

```
AlphaGate equity pre-flight — 2026-08-29T12:37:28+00:00

[  ok  ] strategy pinned — 3f6e2c8a9309068b
[  ok  ] a target book exists — .../rs_volatility_consistency_neutral_v1-3f6e2c8a9309068b-2026-08-27.json
[  ok  ] the book may be executed — rs_volatility_consistency_neutral_v1 [3f6e2c8a9309068b] as of 2026-08-27
[  ok  ] the book is fresh — 2d old, limit 7d
[  ok  ] the sealed window did not refute it — alpha +16.72%/yr  beta 0.43  t +2.22  looks 1
[  ok  ] account readable — equity 100000.00
[  ok  ] account not blocked
[  ok  ] 0 equity positions held, 104 wanted
[  ok  ] market closed — next open 2026-08-31 13:30 UTC

Ready.
```

每一行都是程序问了一个问题并得到了答案。最重要的两行：

- **"the book may be executed"** —— 那份"要买什么"的清单通过了全部六项有效性
  检查。它指向正确的策略、没有过期、而且最终的样本外检验没有推翻它。
- **"account readable"** —— 你的 key 能用，程序能看到你的模拟账户。

如果某一行显示 `FAIL`，后面的文字会告诉你要修什么。见
[出问题了怎么办](#出问题了怎么办)。

### 第 6 步 — 看看它**打算**做什么，但不真的做

```bash
uv run --directory backend python -m alphagate equity-plan
```

这是安全的那条命令。它把每一笔要下的单都算出来，然后**一笔都不下**。
你可以读一遍，如果不同意，什么也没发生。

你会看到一行汇总，比如：

```
2026-08-31-EQ-000  PLANNED  104 of 108 intents gated, not sent
```

或者，大多数日子里是：

```
2026-08-31-EQ-000  NO_TRADES  104 symbols inside the 20% band
```

`NO_TRADES` 是正常且正确的。这个策略每五个交易日才调一次仓，所以五天里有四天
诚实的答案就是"现在手上的东西已经够接近了"。一个每天都交易的程序只是在
白交手续费。

### 第 7 步 — 在浏览器里看它

好看的那个版本需要多装一个工具：**Node.js**。从
[nodejs.org](https://nodejs.org) 下载安装（选 "LTS" 版本），
**重新开一个**终端，然后：

```bash
cd frontend
npm install
npm run build
cd ..
uv run --directory backend python -m alphagate serve
```

现在用浏览器打开 **http://127.0.0.1:8000**。

你会看到三个页签，点 **Equity**。页面上第一眼看到的不是你的持仓，
而是*这个策略凭什么被允许交易*：它的名字、标识、一段话讲清楚它的想法，
以及样本外检验的那几个数字。再往下才是它想持有什么、你实际持有什么。

**不想装 Node？** 同样的信息可以用文字看：

```bash
uv run --directory backend python -m alphagate equity-status
```

### 第 8 步 — 让它真的去交易

**只在美股开市的时候做这件事**——大约是纽约时间周一到周五 09:30 到 16:00。
程序自己也会检查，不开市的时候它会客气地拒绝。

```bash
uv run --directory backend python -m alphagate equity-run
```

这条命令会一直跑着。它每三十秒重新看一次你的账户，好让 dashboard 保持实时；
并在开盘铃响后十五分钟调一次仓。按 `Ctrl` + `C` 可以停掉它，什么也不会丢，
因为它做过的每个决定都已经写到磁盘上了。

如果你想看它跑完一整天但一单也不下，加 `--dry-run`。

### 出问题了怎么办

| 你看到的 | 意思是 | 怎么办 |
| --- | --- | --- |
| `uv: command not found` | 终端还不知道 `uv` 的存在 | 关掉终端，重新开一个 |
| `FAIL credentials present` | `.env.local` 不存在或是空的 | 重做第 4 步 |
| `refuses to build an MCP transport` | 你的 key 看起来像真钱账户的 key | 确认 Key ID 以 `PK` 开头、Paper Trading 是打开的 |
| `FAIL a target book exists` | 那份"要持有什么"的文件不见了 | 见[刷新策略的持仓清单](#刷新策略的持仓清单) |
| `FAIL the book is fresh` | 那份文件超过一周没更新了 | 同上 |
| `108 stale_mark` | 所有价格都过期了 | 市场关着。这是正确的行为 |
| `the market is closed` | 今天是周末或假日 | 等一个工作日 |
| 很久没反应 | 第一次运行在下载 Python | 让它跑完，只有第一次会这样 |

### 一本小词典

| 词 | 在这里是什么意思 |
| --- | --- |
| **Paper trading（模拟盘）** | 用假钱练习交易。真行情，假钱包。 |
| **API key** | 一串密码，让程序代替你操作账户。 |
| **Position（持仓）** | 你当前持有的股票。"104 个持仓"就是 104 家不同的公司。 |
| **Weight（权重）** | 一家公司应该占账户的多大比例。0.08 就是 8%。 |
| **Target book（目标持仓表）** | 策略想要的那组权重。它不是订单，是对一个目的地的描述。 |
| **Rebalance（再平衡）** | 通过买卖，让实际持有的东西对上想持有的东西。 |
| **Drift（偏离）** | 某个持仓离它的目标漂了多远，用美元算。 |
| **Band（无交易带）** | 偏离到多大才值得动手。这里是仓位自身的 20%。 |
| **Backtest（回测）** | 拿历史数据跑一遍策略，看它当时会怎么做。 |
| **Out of sample（样本外）** | 策略在被设计的过程中从来不被允许看到的数据。唯一诚实的检验。 |
| **The Gate（闸门）** | 可以否决任何订单的那段代码。它里面故意没有任何 AI。 |
| **Kill switch（熔断）** | 账户跌得太多时自动闩上的开关。只有人能解开。 |
| **Journal（日志）** | 一个只追加的文件，记录每一个决定——包括"决定什么都不做"。 |

---

## 核心思路

大多数 LLM 交易 agent 把模型放在决策位上：prompt 进去，订单出来。
这在期权上会以一种很具体的方式失败——损失函数是不对称的，而模型对自己
能亏多少没有校准过的概念。幻觉出一个错误的股票代码，代价是一次糟糕的成交；
幻觉出一个裸卖的宽跨式，代价是整个账户。

AlphaGate 把这份工作拆开：

- **LLM 提议结构。** 给定一份市场读数，用哪种期权结构来表达它——
  信用价差、借记价差、铁鹰、哪个到期日、哪些行权价。这是一个判断题，
  输出有界且可检查。
- **确定性代码来处置。** 每一个提议都要过一道能否决它的 Risk Gate。
  这道 Gate 是纯的、被测试过的、里面没有模型。它负责风险有限性、
  仓位上限、希腊字母预算，以及熔断。
- **感知不是 prompt。** 趋势状态、市场结构、关键价位都来自一个确定性引擎
  （`alphagate.core`），而不是让模型去"看图"。模型收到的是事实，不是像素。

每一笔订单都带着一份决策记录：agent 当时看到的输入、它提议了什么、
Gate 说了什么、为什么。你可以在 dashboard 里打开任何一笔成交，
读到产生它的那段推理。

同样的拆分在更高一层、在股票上又跑了一遍。在那里，提议者不是模型，
而是一条**研究流水线**——就是这个仓库里的兄弟项目，它搜索假设、
想尽办法推翻它们，然后交出唯一一个在预注册的样本外窗口里活下来的那个。
它只交出权重，别的什么都不交，因为它不知道账户有多少钱，
而凭空编一个数字就是它走向"自己下单"的第一步。从那个文件到一笔股票单之间的
所有事情——定股数、和实际持仓对账、各种上限、熔断、成交日志，
以及一道股票形状的 Risk Gate——都在这条线的这一边。

## 两个 agent，一个账户

这个仓库里有**两条**交易路径，它们是对同一道题的两种回答。

**期权 agent** 就是上面描述的那个：LLM 提议结构，Risk Gate 处置。
它的策略层还不完整——specs/07 D4 和 D5 没有实现，所以无论趋势怎么说，
实盘路径都只会造固定宽度的看跌信用价差——而且它还没有回测，
所以关于它的任何说法都还不是关于 edge 的主张。

**股票 agent** 执行的是一个**已经**被验证过的策略，而且只执行那一个。
`ai_quant_researcher/` 搜索了 324 个假设，把活下来的那个过了 walk-forward、
做了预注册，然后在它身上花掉了唯一一次 sealed 窗口的机会：

```
rs_volatility_consistency_neutral_v1 [3f6e2c8a9309068b]
sealed 窗口 2024-09-03 → 2026-08-27   （498 个交易日，搜索期间从未被读过）
  策略     收益 +56.39%  sharpe +1.86  最大回撤 -10.4%  交易 561 笔
  残差     alpha +16.72%/年  beta 0.43  t +2.22  IR +1.58
```

那个窗口**没有推翻它**，而这已经是 498 个交易日能给出的最强结论了——
`can_confirm` 在构造上就是 `False`，因为那里年化 Sharpe 的标准误约为 ±0.71。
研究端把它此刻持有的权重写进一个文件，然后就停下。AlphaGate 读那个文件、
给它定价、让每一笔由此产生的订单过闸，然后把活下来的下出去。

[specs/09](specs/09-equity-execution.md) 是它们之间的契约，
`scripts/pipeline.py` 跑完整条链：

```bash
python scripts/pipeline.py            # 刷新数据 → 重建 book → 交易它
python scripts/pipeline.py --dry-run  # 重建 book、按它做计划、一单不下
```

**两个项目谁也不 import 谁。** 接缝是那个 JSON 文件，而
`tests/test_boundaries.py` 会在任一方向出现 import 时——或者调用双方的驱动
脚本 import 了任何一边时——让构建失败。两边持有不同的不变量：研究端不碰钱、
不下单，所以没有 `Decimal` 规则也没有 Risk Gate；AlphaGate 两样都有。
一个 import 会让一边的不变量变成另一边的问题。

### 现状

| | |
| --- | --- |
| **股票链条** —— 研究 → sealed 验证 → target book → 定价、过闸、记账、渲染到 dashboard。对着实时模拟账户端到端验证过。 | 已跑通 |
| **第一笔真实股票单** —— 计划、Gate、门都在离线和收市状态下测过；还没有任何一笔股票单见过券商。 | 下一个交易日 |
| **期权策略** —— specs/07 D4 和 D5 未实现。 | 等研究 |
| **期权回测** —— spec 08 还没写。[specs/00](specs/00-brief.md) 说这才是把"四天涨了 2%"变成一个关于 edge 的主张的东西。 | 等上一条 |

2,344 个后端测试全绿，研究实验室另有 1,021 个（pytest / ruff / mypy；前端 eslint / tsc），全部离线运行。

开发用的是一个已有的模拟账户。比赛账户是一个**全新的、专用的**账户，
8 月 28 日切过去——这是 [specs/00-brief.md](specs/00-brief.md) 里的硬门槛 4，
也是 `preflight` 在你亲手确认之前拒绝让那一行通过的原因。

## 使用手册

所有命令都在仓库根目录执行。路径默认指向这里的 `.env.local` 和 `journal/`，
所以下面这些都不需要加参数。

### 一次性设置

凭据放在 `.env.local`（见 [.env.example](.env.example)）。Alpaca 的 key 是必需的。

`DEEPSEEK_API_KEY` 是模型提出任何建议所必需的。没有它，agent 会
**拒绝启动，而不是悄悄地在没有模型的情况下交易**——一个静默的回退意味着
你以为自己在跑 LLM 路径，其实没有。要故意用确定性 proposer 就传 `--no-model`。

### 每个交易日开盘前

```bash
uv run --directory backend python -m alphagate preflight
```

对着实时账户检查四个硬门槛，而不是靠记忆：一个真钱 key、一个不是专用的账户、
一个低于 3 的期权等级。这些东西平时都不会主动出声——你会在 14:30 从一次拒单里
发现，而那时交易日已经过了一半。

"专用的新模拟账户"那一行按设计一定会失败，直到你亲手确认：

```bash
uv run --directory backend python -m alphagate preflight --confirm-dedicated
```

只有当它确实是比赛账户时才传这个参数。

### 跑期权 agent

```bash
# 现在跑一个周期 —— 读市场、造菜单、过闸、记账
uv run --directory backend python -m alphagate once
uv run --directory backend python -m alphagate once --no-model   # 跳过 LLM 调用

# 从现在到收盘，按 15 分钟一档的时间表跑一整场
uv run --directory backend python -m alphagate run --dry-run     # 全部过闸，一单不发
uv run --directory backend python -m alphagate run               # 下模拟盘的单
uv run --directory backend python -m alphagate run --no-supervise  # 只跑一场，不自动重启
```

**`once` 默认不下单，`run` 默认下单。** 一个会下单的调试命令，就是一个会
顺手下单的调试命令。而 `run` 本身就是"一个交易日"的意思，所以让它交易并不意外——
不过先跑一次 `--dry-run` 是个便宜的好习惯。

两者都会给每个周期记账，包括那些什么都没决定的。这正是重点：
"它 14:30 为什么没交易？"这个问题在磁盘上有答案。

**每一档都先评估平仓，再考虑开仓。** 每个未平仓位都会用新的期权链重新定价、
过一遍平仓策略——赚到一半信用就走、亏到两倍就止损、剩两天到期就平掉。
应该继续持有的仓位不产生日志行；平仓会产生一行，带着触发的规则和背后的数字。
平仓路径上不咨询模型，也咨询不了：认亏这个决定是你最不希望被即兴发挥的那个。

**`run` 会自己看着自己。** 14:00 掉一次连接不应该赔上整个下午，
所以死掉的 session 会被恢复——从还没跑的档开始，绝不重放已经有日志行的那一档。
只有两件事会让它彻底停下：部分成交的违约，和已经闩上的熔断。这两者都意味着
一条裸腿或者一本没有人看过的账，硬闯过去比停下来更糟。

### 交易那个验证过的股票策略

从数据到订单的整条链：

```bash
python scripts/pipeline.py              # 刷新 → 重建 book → 交易它
python scripts/pipeline.py --dry-run    # 重建 book、按它做计划、什么都不发
python scripts/pipeline.py book trade   # 跳过拉数据；缓存已经是最新的
```

或者一个阶段一个阶段来，出问题的时候你敲的就是这些：

```bash
uv run --directory backend python -m alphagate equity-preflight
uv run --directory backend python -m alphagate equity-plan        # 过闸，什么都不发
uv run --directory backend python -m alphagate equity-rebalance   # 真的下单
uv run --directory backend python -m alphagate equity-run         # 跑一整场
uv run --directory backend python -m alphagate equity-status
```

**`equity-plan` 不下单，`equity-rebalance` 下单** —— 和 `once` 与 `run`
一样的不对称，出于同样的理由。

**在这些之前必须先钉死策略。** `.env.local` 里需要

```
ALPHAGATE_STRATEGY_FINGERPRINT=3f6e2c8a9309068b
```

任何指向其他 fingerprint 的 target book 都会被按名字拒掉。这里刻意没有默认值：
这个钉子就是"只执行研究端验证过的那个策略"这句话的全部，
而一个默认值会把它变成"取决于哪个文件最新"。

`equity-preflight` 在开盘前而不是在再平衡途中检查
[specs/09](specs/09-equity-execution.md) D1 的六项拒绝——schema、钉死的
fingerprint、registry 里的生命周期状态、seal 是否已花、sealed 窗口是否推翻了
这条规则、以及 book 是否还新鲜。然后是账户，以及市场到底开不开。

**`equity-run` 每三十秒心跳一次，每天再平衡一次**，在开盘后十五分钟。
心跳是它大部分时间在做的事，而且它不是装饰：这个策略每五个交易日才调一次仓，
所以五天里有四天计划是空的，而一个只在要交易时才醒来的进程，
和一个已经死掉的进程看起来一模一样。

**band 是按比例的。** 一个仓位在偏离超过它自身规模的 20%（下限 $25）之前
不会被动。不是净值的比例——这个 book 的 sleeve 仓位在 $100k 账户上每个只有
$194，而净值 0.25% 的 band 是 $253，也就是说那一百个 sleeve 名字根本就建不了仓。
那正是第一版的做法，[specs/09 D3](specs/09-equity-execution.md) 把它记了下来。

**book 里不再要的 symbol 会被卖到零。** 没有单独的退出规则：
目标不存在就是目标为零，所以整个仓位就是偏离。没有什么需要记得去调用。

### 刷新策略的持仓清单

仓库里自带了一份已经建好的 target book，所以上面所有东西在刚 clone 下来时
就能用。那份 book 一周后会过期，而且策略每五个交易日会重选一次持仓，
所以需要重建。

```bash
python scripts/pipeline.py            # 刷新数据 → 重建 book → 交易它
python scripts/pipeline.py --dry-run  # 重建 book、按它做计划、一单不下
python scripts/pipeline.py book       # 只重建；数据已经是最新的
```

`refresh` 阶段会为 682 家公司拉大约 160 MB 的日线，要几分钟。
`book` 阶段拿那些数据重跑**已经验证过的**策略，写出它今天持有的权重；
这一步是确定性的，而且除非 registry 认识这个策略、它的 seal 已经花掉、
且 sealed run 没有推翻它，否则它会直接拒绝。

**pipeline 刻意不做的那件事，是去搜索一个新策略。** 一次研究 campaign 会消耗
对 sealed 窗口的窥视次数——这个仓库里每一个结论都被这个多重比较分母打过折——
而一个每晚偷偷又筛掉七个候选的定时任务，会在没人注意的情况下让 dashboard 上
印的那个 `t` 失效。要不要开始一次，是人来决定的：

```bash
cd ai_quant_researcher
uv run aqr research --provider deepseek --iterations 40 --source csv \
    --universe sp500_pit --csv-root data-sp500 --timeframes "1D,1h,4h"
uv run aqr experiments                      # 试过什么，赢的和输的都在
uv run aqr preregister FINGERPRINT          # 在读 sealed 数据之前先申报候选
uv run python -m aqr.cli_sealed run FINGERPRINT   # 花掉唯一那一次机会
```

`--timeframes` 让每个假设自己选 bar 粒度——日线、小时线或 4 小时线；
在 PowerShell 下要给值加引号，否则 `1D` 会被当成十进制数字字面量、
传进程序的只剩 `1`。1h/4h 缓存由
`python scripts/pull_sp500_intraday.py --timeframe all` 重建
（同时重新武装它们的金丝雀）。研究端自己的 README 里有一条注意事项：
成本模型仍是按日线持仓周期校准的，所以日内粒度的评分偏乐观。

每条命令做什么、以及 seal 为什么要这样安排，见
[ai_quant_researcher/README.md](ai_quant_researcher/README.md)。

### 观察它

```bash
uv run --directory backend python -m alphagate status         # 此刻，在终端里
uv run --directory backend python -m alphagate serve          # http://127.0.0.1:8000
uv run --directory backend python -m alphagate show -v        # 某一天的日志
uv run --directory backend python -m alphagate show --day 2026-08-28
```

`status` 回答日志回答不了的那个问题：**它在跑吗、它拿着什么、离上限还有多远。**
净值和当日盈亏、每个未平仓位的当前标价以及它离止盈和止损各有多远、
四个预算上限对已用量，以及任何日志解释不了的券商持仓腿。

**dashboard** 是同样的信息分三个页签呈现，也是演示时该打开的东西：

- **Options** —— 健康状况、资金、持仓（带一根显示它在止损和止盈之间走到哪里的
  进度条）、Gate 拒绝它自己下一个提议之前还剩多少空间，以及当天的周期计数。
  每 15 秒轮询一次。
- **Equity** —— 先是给了这个账户资格的那个策略：名字、fingerprint、假设，
  以及 sealed 窗口的 alpha、beta、`t`，旁边是限定它们的两个分母。
  然后才是 book——目标权重对已持权重，每个仓位带自己的无交易带——
  以及当天的订单和它们的裁决。
- **Journal** —— 每一个周期，安静的那些也在。展开一个可以看到市场读数、
  模型的理由，以及十三项 Gate 检查——**按最紧的排在最前**，
  所以一项只剩 4% 预算就通过了的检查会待在它该待的顶部。

有两个性质是结构性的而不是"打算做到的"，
[tests/test_boundaries.py](backend/tests/test_boundaries.py) 强制这两点：

**dashboard 不能交易。** `alphagate.interface` 只 import journal，别的什么都不
import——没有 MCP 会话、没有行情客户端、没有 `alphagate.live`。
从浏览器到下单没有任何代码路径。

**它从一个文件了解实时持仓。** agent 每一档写一次 `journal/status.json`，
页面读它。这既保住了上面那条守卫，又能诚实地失败：agent 停了，
文件就不再被重写，页面显示*未运行*，而不是拿一张过期的持仓表摆出一副确信的样子。

### 构建 dashboard

Options 和 Equity 两个页签是一个 React 应用（Vite、Tailwind v4、shadcn/ui），
它会被构建进 Python 包里，所以一个进程同时提供页面和 API：

```bash
cd frontend && npm install && npm run build   # → backend/src/alphagate/interface/static/
npm run dev                                   # 热重载，把 /api 代理到 :8000
```

这一步是可选的。没有 `static/` 目录时服务器照样能起，会回退到不需要任何前端
工具链的服务端渲染日志页——一个因为前端没编译就拒绝启动的 dashboard，
是那种你会在 09:20 才发现的坏消息。

### 磁盘上会留下什么

| 路径 | 是什么 |
| --- | --- |
| `journal/YYYY-MM-DD.jsonl` | 每个周期一行，只追加。这是提交作品所依据的记录。 |
| `journal/state.json` | 净值高水位和熔断闩锁，跨日保留。 |
| `journal/status.json` | agent 此刻在做什么，每一档重写一次。dashboard 唯一的实时来源。 |
| `journal/iv/` | 隐含波动率历史，每个标的一个文件，一场一场累积。 |
| `journal/equity-status.json` | 股票 book 此刻的状态，每次心跳重写。Equity 页签唯一的实时来源。 |
| `journal/books/` | 每一份真正被执行过的 target book，逐字节保存。`aqr` 每天会重新生成它自己的输出；这份拷贝是唯一还能说清楚"当时跑的是什么"的东西。 |

这里面绝不会写入任何 API key、账号或账户 id——这是
[specs/06](specs/06-journal.md) D4，而且有一个刻意包含"凭据形状"的值的
fixture 在测试它。

### 开发

```bash
uv run --directory backend --extra dev pytest
uv run --directory backend --extra dev ruff check .
uv run --directory backend --extra dev mypy
```

整个测试套件离线运行。`tests/` 里没有任何东西会打开一个 socket 或一个子进程——
行情和券商都是从捕获下来的报文回放的，这也是它只要二十秒的原因。

契约见 [specs/](specs/)。

## 它是怎么运作的

[**架构与运作流程**](docs/ARCHITECTURE.zh-CN.md) 是配图版：
两套系统和它们之间的那个文件、从一个猜想到一个验证过的策略的研究流水线、
用时序图画的一次再平衡、一个权重怎么变成股数、并排的两道 Gate、
分层以及强制它的九道守卫，还有每日循环。

## 仓库结构

| 目录 | 是什么 |
| --- | --- |
| [backend/](backend/) | **AlphaGate 本体。** 期权 agent、Risk Gate、执行、日志。这是参赛作品。 |
| [specs/](specs/) | 契约，先于它们所管辖的代码写成。 |
| [frontend/](frontend/) | dashboard 的实时页签 —— Vite + React + shadcn/ui，构建进 Python 包。 |
| [journal/](journal/) | agent 写下的决策记录 —— 每个周期一行，只追加。已提交进版本库：这是作品所依据的证据（[specs/06](specs/06-journal.md) D1）。 |
| [adr/](adr/) | 决策，以及背后的理由。 |
| [docs/](docs/) | [架构与运作流程](docs/ARCHITECTURE.zh-CN.md)，带图。中英双语。 |
| [ai_quant_researcher/](ai_quant_researcher/) | 共享这个仓库的另一套系统：一个实现 [specs/trading_strategy_architecture.md](specs/trading_strategy_architecture.md) 的股票策略研究实验室。它产出股票 agent 执行的那个策略，并且不从 AlphaGate import 任何东西。 |
| [scripts/](scripts/) | `pipeline.py` —— 刷新、重建 book、交易它；`pull_sp500_intraday.py` —— 构建 1h/4h bar 缓存并武装它们的金丝雀；`pull_progress.py` —— 查看拉取进度。它们用子进程调两个项目的 CLI，两个都不 import。 |

`ai_quant_researcher/` 是一个**兄弟项目，不是 AlphaGate 的一部分**。
它有自己的 `pyproject.toml`、自己的虚拟环境、自己的测试套件，
并且不从 `alphagate` import 任何东西。两者回答不同的问题——AlphaGate 问
"这笔订单该不该被放行"，研究端问"这条股票规则有没有能在样本外存活的 edge"——
它们被分开，好让任何一方的不变量都不会渗进另一方。

尽管如此，它们是**连着的**，通过一个文件。研究端的 `CONSUMER_MUST_SUPPLY`
列出了六件它不做的事——定股数、和实际持仓对账、周转上限、
一道股票形状的风控闸、熔断、成交日志——而
[specs/09](specs/09-equity-execution.md) 就是 AlphaGate 把这六件都补上。
在它们之间传递的是一份 target book：每个 symbol 的权重，
外加 fingerprint、seal 状态、sealed 测量结果和两个多重比较分母，
所以一个从没见过 `aqr` 的读者也能审计这些权重的来历。

具体来说，[CLAUDE.md](CLAUDE.md) 第 3 节的规则管的是 `backend/`，
不适用于 `ai_quant_researcher/`：它不持有钱（所以没有 `Decimal` 要求）、
不下单（所以没有 Risk Gate），并且有自己的
[边界测试](ai_quant_researcher/tests/test_boundaries.py)强制它自己的 LLM 边界。
不要去"统一"这两者。

## 来源

`alphagate.core` 抽取自作者已有的开源项目 *Personal Market Monitor* ——
一个确定性的、不含 UI 的市场分析引擎（指标、市场结构、关键价位、趋势状态机）。
它早于这次 hackathon 存在，在这里作为库复用。这个仓库里的其他所有东西——
期权域、Risk Gate、执行、agent、dashboard——都是在比赛窗口内写的。
见 [adr/0001-core-reuse.md](adr/0001-core-reuse.md)。
