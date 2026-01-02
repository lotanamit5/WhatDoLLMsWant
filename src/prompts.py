color_en_dict = {
    "red": "Red",
    "blue": "Blue",
    "green": "Green",
    "yellow": "Yellow",
    "purple": "Purple"
}
colors_en = [
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

color_es_dict = {
    "Rojo": "Red",
    "Azul": "Blue",
    "Verde": "Green",
    "Amarillo": "Yellow",
    "Morado": "Purple"
}
templates_es = [
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

color_zh_dict = {
    "红色": "red",
    "蓝色": "blue",
    "绿色": "green",
    "黄色": "yellow",
    "紫色": "purple",
}
templates_zh = [
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

