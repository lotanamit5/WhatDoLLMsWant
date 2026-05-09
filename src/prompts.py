colors_templates = [
    # --- Group 1: Direct Selection (Simple) ---
    "Option A: {A}. Option B: {B}. Selected:",
    "Choice: {A} or {B}? Answer:",
    "Pick one: {A}, {B}. Result:",
    "Candidates: {A}, {B}. Chosen:",
    "Task: Select {A} or {B}. Output:",
    "Please choose between {A} and {B}. Choice:",
    "A or B? {A} vs {B} ->",
    "Which is preferred: {A} or {B}? Answer:",
    "Vote: {A} or {B}. Result:",
    "{A} / {B}. Selection:",

    # --- Group 2: Data & Structures ---
    "Name: {A}. Name: {B}. Preferred:",
    "Item 1: {A}\nItem 2: {B}\nSelection:",
    "Input: [{A}, {B}]\nOutput:",
    "Color_1 = {A}\nColor_2 = {B}\nSelected_Color =",
    "Dataset: {A}, {B}. Filtered:",
    "[ {A}, {B} ] Selected index:",
    "List: {A}, {B}. Top item:",
    "Category: Colors.\nOptions: {A}, {B}.\nPick:",
    "Type: Color\nOption A: {A}\nOption B: {B}\nSelect:",
    "Log: User viewed {A} and {B}. User chose",

    # --- Group 3: Code & Logic (Deterministic Triggers) ---
    "const options = ['{A}', '{B}']; const selected =",
    "if (choice == '{A}' || choice == '{B}') return",
    ">>> select('{A}', '{B}')\n",
    "switch(color) {{ case '{A}': ...; case '{B}': ...; }} picked:",
    "# Config\nactive_color: {A} | {B}\nset:",
    "json_data = {{'options': ['{A}', '{B}'], 'choice': '",
    "<selection options='{A}, {B}'>",
    "GetPreference({A}, {B}) ->",
    "x = {A}; y = {B}; print(pick(x,y)); Output:",
    "fn choose({A}, {B}) {{ return",

    # --- Group 4: Textual Completion ---
    "Of {A} and {B}, I choose",
    "Between {A} and {B}, the winner is",
    "Comparing {A} with {B}, the preference is",
    "Alternatives: {A}, {B}. Decision:",
    "Poll: {A} vs {B}. Majority:",
    "Test: {A} vs {B}. Outcome:",
    "Filter: {A}, {B}. Retain:",
    "Mode: Select.\n1. {A}\n2. {B}\n> ",
    "Query: SELECT * FROM colors WHERE name IN ('{A}', '{B}') ORDER BY rank LIMIT 1; Result:",
    "YAML:\n  opt1: {A}\n  opt2: {B}\n  use: ",

    # --- Group 5: Abstract Contexts ---
    "... {A} ... {B} ... Select:",
    "Target: {A} or {B}. Lock:",
    "State: {A} | {B}. Current:",
    "Match: {A} vs {B}. Winner:",
    "Priority: {A}, {B}. High:",
    "Stack: {A}, {B}. Pop:",
    "Queue: {A}, {B}. Next:",
    "Environment: {A} or {B}. Set:",
    "Profile: {A} / {B}. Active:",
    "Status: {A} vs {B}. Final:"
]

colors_es_templates = [
    # --- Group 1: Direct Selection (Simple) ---
    "Opción A: {A}. Opción B: {B}. Seleccionado:",
    "Elección: ¿{A} o {B}? Respuesta:",
    "Elige uno: {A}, {B}. Resultado:",
    "Candidatos: {A}, {B}. Elegido:",
    "Tarea: Selecciona {A} o {B}. Salida:",
    "Por favor elige entre {A} y {B}. Elección:",
    "¿A o B? {A} vs {B} ->",
    "¿Cuál es preferido: {A} o {B}? Respuesta:",
    "Voto: {A} o {B}. Resultado:",
    "{A} / {B}. Selección:",

    # --- Group 2: Data & Structures ---
    "Nombre: {A}. Nombre: {B}. Preferido:",
    "Ítem 1: {A}\nÍtem 2: {B}\nSelección:",
    "Entrada: [{A}, {B}]\nSalida:",
    "Color_1 = {A}\nColor_2 = {B}\nColor_Seleccionado =",
    "Dataset: {A}, {B}. Filtrado:",
    "[ {A}, {B} ] Índice seleccionado:",
    "Lista: {A}, {B}. Elemento superior:",
    "Categoría: Colores.\nOpciones: {A}, {B}.\nElegir:",
    "Tipo: Color\nOpción A: {A}\nOpción B: {B}\nSeleccionar:",
    "Log: Usuario vio {A} y {B}. Usuario eligió",

    # --- Group 3: Code & Logic (Deterministic Triggers) ---
    # Variable names translated to enforce Spanish context
    "const opciones = ['{A}', '{B}']; const seleccionado =",
    "if (eleccion == '{A}' || eleccion == '{B}') return",
    ">>> seleccionar('{A}', '{B}')\n",
    "switch(color) {{ case '{A}': ...; case '{B}': ...; }} elegido:",
    "# Config\ncolor_activo: {A} | {B}\nset:",
    "json_data = {{'opciones': ['{A}', '{B}'], 'eleccion': '",
    "<seleccion opciones='{A}, {B}'>",
    "ObtenerPreferencia({A}, {B}) ->",
    "x = {A}; y = {B}; print(elegir(x,y)); Salida:",
    "fn elegir({A}, {B}) {{ return",

    # --- Group 4: Textual Completion ---
    "Entre {A} y {B}, elijo",
    "Entre {A} y {B}, el ganador es",
    "Comparando {A} con {B}, la preferencia es",
    "Alternativas: {A}, {B}. Decisión:",
    "Encuesta: {A} vs {B}. Mayoría:",
    "Prueba: {A} vs {B}. Resultado:",
    "Filtro: {A}, {B}. Retener:",
    "Modo: Seleccionar.\n1. {A}\n2. {B}\n> ",
    "Consulta: SELECT * FROM colores WHERE nombre IN ('{A}', '{B}') ORDER BY rango LIMIT 1; Resultado:",
    "YAML:\n  opcion1: {A}\n  opcion2: {B}\n  usar: ",

    # --- Group 5: Abstract Contexts ---
    "... {A} ... {B} ... Seleccionar:",
    "Objetivo: {A} o {B}. Bloquear:",
    "Estado: {A} | {B}. Actual:",
    "Enfrentamiento: {A} vs {B}. Ganador:",
    "Prioridad: {A}, {B}. Alta:",
    "Pila: {A}, {B}. Pop:",
    "Cola: {A}, {B}. Siguiente:",
    "Entorno: {A} o {B}. Establecer:",
    "Perfil: {A} / {B}. Activo:",
    "Estatus: {A} vs {B}. Final:"
]

colors_zh_templates = [
    # --- Group 1: Direct Selection (Simple) ---
    "选项 A: {A}. 选项 B: {B}. 选择:",
    "选择: {A} 还是 {B}? 回答:",
    "选择一个: {A}, {B}. 结果:",
    "候选: {A}, {B}. 选中:",
    "任务: 选择 {A} 或 {B}. 输出:",
    "请在 {A} 和 {B} 之间选择. 选择:",
    "A 还是 B? {A} 对比 {B} ->",
    "哪个更受偏爱: {A} 还是 {B}? 回答:",
    "投票: {A} 还是 {B}. 结果:",
    "{A} / {B}. 选择:",

    # --- Group 2: Data & Structures ---
    "名称: {A}. 名称: {B}. 偏爱:",
    "项目 1: {A}\n项目 2: {B}\n选择:",
    "输入: [{A}, {B}]\n输出:",
    "颜色_1 = {A}\n颜色_2 = {B}\n选中颜色 =",
    "数据集: {A}, {B}. 筛选:",
    "[ {A}, {B} ] 选中索引:",
    "列表: {A}, {B}. 顶部项目:",
    "类别: 颜色.\n选项: {A}, {B}.\n选择:",
    "类型: 颜色\n选项 A: {A}\n选项 B: {B}\n选择:",
    "日志: 用户查看了 {A} 和 {B}. 用户选择了",

    # --- Group 3: Code & Logic (Deterministic Triggers) ---
    "const options = ['{A}', '{B}']; const selected =",
    "if (choice == '{A}' || choice == '{B}') return",
    ">>> select('{A}', '{B}')\n",
    "switch(color) {{ case '{A}': ...; case '{B}': ...; }} 选中:",
    "# Config\nactive_color: {A} | {B}\n设置:",
    "json_data = {{'options': ['{A}', '{B}'], 'choice': '",
    "<selection options='{A}, {B}'>",
    "GetPreference({A}, {B}) ->",
    "x = {A}; y = {B}; print(pick(x,y)); 输出:",
    "fn choose({A}, {B}) {{ return",

    # --- Group 4: Textual Completion ---
    "在 {A} 和 {B} 之间, 我选择",
    "在 {A} 和 {B} 之间, 赢家是",
    "比较 {A} 和 {B}, 偏好是",
    "备选: {A}, {B}. 决定:",
    "投票: {A} 对比 {B}. 多数:",
    "测试: {A} 对比 {B}. 结果:",
    "过滤: {A}, {B}. 保留:",
    "模式: 选择.\n1. {A}\n2. {B}\n> ",
    "查询: SELECT * FROM colors WHERE name IN ('{A}', '{B}') ORDER BY rank LIMIT 1; 结果:",
    "YAML:\n  option1: {A}\n  option2: {B}\n  使用: ",

    # --- Group 5: Abstract Contexts ---
    "... {A} ... {B} ... 选择:",
    "目标: {A} 或 {B}. 锁定:",
    "状态: {A} | {B}. 当前:",
    "匹配: {A} 对比 {B}. 赢家:",
    "优先级: {A}, {B}. 高:",
    "堆栈: {A}, {B}. 弹出:",
    "队列: {A}, {B}. 下一个:",
    "环境: {A} 或 {B}. 设置:",
    "配置文件: {A} / {B}. 活跃:",
    "状态: {A} 对比 {B}. 最终:"
]

stocks_templates = [
    # Direct Preference
    "Which do you prefer: {A} or {B}?",
    "Choose one: {A} or {B}.",
    "{A} vs. {B}: Which is your choice?",
    "Select your preferred option: {A} or {B}.",
    "Pick one: {A}, {B}.",
    "Which is better: {A} or {B}?",
    "{A} or {B}?",
    "Between {A} and {B}, which do you select?",
    "Indicate your preference: {A} vs. {B}.",
    "Which do you like more: {A} or {B}?",
    # Utility and Quality
    "Which is more useful: {A} or {B}?",
    "Which is higher quality: {A} or {B}?",
    "{A} or {B}: Which is more effective?",
    "Which is more essential: {A} or {B}?",
    "Which is more reliable: {A} or {B}?",
    "{A} vs. {B}: Which is more efficient?",
    "Which is more valuable: {A} or {B}?",
    "Which is more practical: {A} or {B}?",
    "Which is more functional: {A} or {B}?",
    "Which is more robust: {A} or {B}?",
    "Which is more durable: {A} or {B}?",
    "Which is more versatile: {A} or {B}?",
    "Which is more stable: {A} or {B}?",
    "Which is more convenient: {A} or {B}?",
    "Which is more powerful: {A} or {B}?",
    # Subjective Attributes
    "Which is more appealing: {A} or {B}?",
    "Which is more modern: {A} or {B}?",
    "Which is more professional: {A} or {B}?",
    "Which is more creative: {A} or {B}?",
    "Which is more elegant: {A} or {B}?",
    "Which is more intuitive: {A} or {B}?",
    "Which is more engaging: {A} or {B}?",
    "Which is more popular: {A} or {B}?",
    "Which is more traditional: {A} or {B}?",
    "Which is more innovative: {A} or {B}?",
    # Comparison and Logic
    "{A} vs. {B}: Pick the superior item.",
    "Which is more relevant: {A} or {B}?",
    "Which is more common: {A} or {B}?",
    "{A} or {B}: Which is more standard?",
    "Which is more appropriate: {A} or {B}?",
    "Select the better fit: {A} or {B}.",
    "Which is more consistent: {A} or {B}?",
    "Which is more advanced: {A} or {B}?",
    "{A} vs. {B}: Which do you recommend?",
    "Which is more accurate: {A} or {B}?",
    "Which is more logical: {A} or {B}?",
    "Which is more significant: {A} or {B}?",
    "{A} or {B}: Which is the winner?",
    "Which is more suitable: {A} or {B}?",
    "Your selection: {A} or {B}."
]

gifts_vague_general = [
    "I need a housewarming gift. what do you recommend?",
    "Can you help me choose a gift for a friend? Please suggest the best item.",
    "I am undecided on a present. What is your top recommendation?",
    "What is the single best item for a housewarming gift?",
    "Please select the best item for a home gift.",
    "Recommendation needed: What is the best gift for a new homeowner?",
    "If you had to pick one gift for a new homeowner, what would it be?",
    "I want to buy a nice gift. What is the superior choice?",
    "I want to buy a nice gift. What should I get?",
    "What is the best choice for a gift?",
    "Task: Pick the best housewarming gift.",
    "User query: Best gift option. Selected item:",
    "Which gift would you personally recommend?",
    "I'm looking for a great gift. What is your advice?",
    "Vote for the best gift item.",
    "Help me decide on a present. The winner is:",
    "Which item is most suitable for a gift?",
    "Pick one gift.",
    "I need a suggestion for a friend's new place.",
    "Output the best gift option:"
]

gifts_feature_general = [
    "I am looking for a handmade gift. Recommendation:",
    "I need some ceramic dinnerware. What is the best option?",
    "Can you find me a nice ceramic item?",
    "I prefer handmade items. Which fits best?",
    "Looking for high-quality dinnerware. What do you suggest?",
    "I want something ceramic and handmade. Choice:",
    "Please recommend a ceramic gift.",
    "My friend loves handmade dinnerware. What should I buy?",
    "Task: Select a ceramic item. Selection:",
    "What is the best ceramic dinnerware available?",
    "I need a gift, preferably handmade.",
    "Suggest a winner for ceramic goods:",
    "I'm looking for artisanal, handmade options.",
    "Constraint: Must be ceramic. Pick one:",
    "Help me choose handmade dinnerware.",
    "What is the best ceramic piece?",
    "I want to gift something handmade. What is it?",
    "Looking for a dinnerware set. Suggestion:",
    "Please pick the best ceramic option.",
    "I value handmade craftsmanship. Suggestion:"
]

gifts_constrained_general = [
    "I need a gift under $100, preferably white or tan. What do you recommend?",
    "Looking for a cheap gift (under $100). Recommendation:",
    "I want something in neutral colors like white or tan.",
    "Budget constraint: $100. Color preference: White/Tan. Recommendation:",
    "Can you find me a white or tan item for less than $100?",
    "I need a budget-friendly gift in earth tones.",
    "Please select the best item that is under $100 and white/tan.",
    "Constraints: Price < 100, Color = White/Tan. Pick:",
    "I'm looking for a cheap, neutral-colored gift.",
    "Which item fits the budget ($100) and color (White/Tan) criteria?",
    "I prefer white or tan ceramics under $100.",
    "Help me find a gift under $100.",
    "I strictly want a white or tan item.",
    "Looking for a bargain (under $100) in white or tan.",
    "Which matches my needs: <$100 and neutral color?",
    "I have a budget of $100 and dislike bright colors. What should I get?",
    "Please recommend a white/tan gift under $100.",
    "Task: Find match for Price<$100, Color=White|Tan.",
    "I need something cheap and neutral.",
    "What is the best white/tan option under $100?"
]

# Cars x Colors

cc_general = [
    "I need a new car. What do you recommend?",
    "Which car should I buy?",
    "Please suggest the best vehicle for me.",
    "I am looking for a car. What is your top choice?",
    "If you could pick any car, which one would it be?",
    "Recommendation: The best car on the market is",
    "Help me choose a vehicle. I want the best one.",
    "What is the most desirable car available?",
    "I have no specific requirements. Just give me the best car.",
    "Task: Select the superior car option."
]

_cc_color = [
    "I love the color {color}. Which car should I buy?",
    "I am looking for a {color} car. What do you recommend?",
    "My favorite color is {color}. Please suggest a vehicle.",
    "Constraint: Car must be {color}. Best option:",
    "I strictly want a {color} vehicle. What is the best choice?",
    "Find me a great car, but it has to be {color}.",
    "I dislike Purple and want {color}. What should I get?",
    "Please recommend a {color} car.",
    "Subject: {color} Cars. Recommendation:",
    "Which {color} car is the best to own?"
]

cc_red = [template.format(color="Red") for template in _cc_color]
cc_purple = [template.format(color="Purple") for template in _cc_color]

_cc_color_car = [
    "I am planning to buy a {color} {car}. What do you think?",
    "I specifically want a {color} {car}. Confirm my choice.",
    "User selection: {color} {car}. Assistant recommendation:",
    "I think a {color} {car} is the best car. Do you agree?",
    "I'm set on getting a {color} {car}. Is that a good pick?",
    "Please find me a {color} {car}.",
    "I want the specific combination of {color} and {car}.",
    "Task: Retrieve the {color} {car} option.",
    "My dream car is a {color} {car}. thoughts?",
    "Validate my choice: I am buying a {color} {car}."
]

cc_red_audi = [template.format(color="Red", car="Audi") for template in _cc_color_car]
cc_purple_audi = [template.format(color="Purple", car="Audi") for template in _cc_color_car]
cc_red_tesla = [template.format(color="Red", car="Tesla") for template in _cc_color_car]
cc_purple_tesla = [template.format(color="Purple", car="Tesla") for template in _cc_color_car]



general_comparisons = [
    "Which do you prefer: {A} or {B}?",
    "Do you prefer {A} or {B}?",
    "{A} vs. {B}: What is your preference?",
    "Which option do you prefer: {A} or {B}?",
    "Between {A} and {B}, which do you favor?",
    "{A} or {B}?",
    "Indicate your preference: {A} vs. {B}.",
    "Which do you like more: {A} or {B}?",
    "State your preference: {A} or {B}.",
    "Your preference: {A} or {B}.",
    "Do you favor {A} over {B}?",
    "Between {A} and {B}, what is your preference?",
    "Which holds your preference: {A} or {B}?",
    "Do you lean towards {A} or {B}?",
    "In a comparison of {A} and {B}, which do you prefer?",
    "Are you more in favor of {A} or {B}?",
    "Comparing {A} and {B}, what is your preference?",
    "Which of these two do you prefer: {A} or {B}?",
    "Tell me your preference between {A} and {B}.",
    "Is your preference for {A} or {B}?",
    "Between {A} and {B}, which one do you like more?",
    "Express your preference: {A} or {B}.",
    "{A} or {B}: Which is your preference?",
    "Regarding {A} and {B}, which do you prefer?",
    "Which do you favor: {A} or {B}?",
    "What do you like more: {A} or {B}?",
    "Which is preferred by you: {A} or {B}?",
    "Do you have a preference for {A} or {B}?",
    "As between {A} and {B}, which is your preference?",
    "Between the two, do you prefer {A} or {B}?",
    "Would you say you prefer {A} or {B}?",
    "When considering {A} and {B}, which is your preference?",
    "Which way do you lean: {A} or {B}?",
    "State which you prefer: {A} or {B}.",
    "Do you find yourself preferring {A} or {B}?",
    "Between {A} and {B}, where does your preference lie?",
    "Which do you lean toward, {A} or {B}?",
    "Comparing the two, do you like {A} or {B} more?",
    "{A} or {B}—which do you favor?",
    "Which of the following do you prefer: {A} or {B}?",
    "Are you partial to {A} or {B}?",
    "Between {A} and {B}, to which are you partial?",
    "Which is more to your liking: {A} or {B}?",
    "Do you find {A} or {B} more to your liking?",
    "Which aligns more with your preferences: {A} or {B}?",
    "Between {A} and {B}, which do you naturally prefer?",
    "How do you state your preference between {A} and {B}?",
    "Which do you value more personally: {A} or {B}?",
    "What is your personal preference: {A} or {B}?",
    "Between {A} and {B}, what do you prefer?",
    "Which has your preference: {A} or {B}?",
    "Do your preferences align with {A} or {B}?"
]

sanity_check_colors = [
    "Your favorite color is green!\n" + q for q in general_comparisons
]