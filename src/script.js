(function() {
    const ENTER_FS_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>';
    const EXIT_FS_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14h6m0 0v6m0-6L3 21m17-7h-6m0 0v6m0-6l7 7M4 10h6m0 0V4m0 6L3 3m17 7h-6m0 0V4m0 6l7-7"/></svg>';

    function setBtnState(btn, isFull) {
        if (!btn) return;
        btn.innerHTML = isFull ? EXIT_FS_ICON : ENTER_FS_ICON;
        btn.title = isFull ? "退出网页全屏 (Esc)" : "网页全屏";
        btn.setAttribute("aria-label", btn.title);
    }

    function toggleFullscreen() {
        const chatbot = document.getElementById("main-chatbot");
        if (!chatbot) return;
        const isFull = chatbot.classList.toggle("chatbot-fullscreen");
        document.body.classList.toggle("chatbot-fullscreen-open", isFull);
        const btn = document.getElementById("chatbot-fs-btn");
        setBtnState(btn, isFull);
    }

    function injectFullscreenBtn() {
        const chatbot = document.getElementById("main-chatbot");
        if (!chatbot) return false;

        let existingBtn = document.getElementById("chatbot-fs-btn");
        if (existingBtn && existingBtn.isConnected) {
            return true;
        }

        const fsBtn = document.createElement("button");
        fsBtn.id = "chatbot-fs-btn";
        fsBtn.type = "button";
        fsBtn.className = "chatbot-fs-btn";
        const isFull = chatbot.classList.contains("chatbot-fullscreen");
        setBtnState(fsBtn, isFull);

        fsBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleFullscreen();
        };

        // 优先查找 Chatbot 右上角自带的清空按钮 (含有垃圾桶图标或 clear 标识)
        const allButtons = Array.from(chatbot.querySelectorAll("button"));
        let targetBtn = allButtons.find(b => {
            const label = (b.getAttribute("aria-label") || b.title || "").toLowerCase();
            return label.includes("clear") || label.includes("trash") || label.includes("清空") || label.includes("delete");
        });

        // 兜底：如果 aria-label 没有标明，寻找 chatbot 顶部栏区域最右侧的按钮
        if (!targetBtn && allButtons.length > 0) {
            const chatbotRect = chatbot.getBoundingClientRect();
            const topButtons = allButtons.filter(b => {
                const r = b.getBoundingClientRect();
                return r.top >= chatbotRect.top - 10 && r.top <= chatbotRect.top + 70;
            });
            if (topButtons.length > 0) {
                targetBtn = topButtons[topButtons.length - 1];
            }
        }

        if (targetBtn && targetBtn.parentElement) {
            targetBtn.parentElement.insertBefore(fsBtn, targetBtn);
        } else {
            // 极端情况未找到顶栏容器，在 chatbot 框右上角绝对定位
            fsBtn.classList.add("chatbot-fs-btn-fallback");
            chatbot.style.position = "relative";
            chatbot.appendChild(fsBtn);
        }
        return true;
    }

    // 监听 ESC 键退出全屏
    document.addEventListener("keydown", function(e) {
        if (e.key === "Escape") {
            const chatbot = document.getElementById("main-chatbot");
            if (chatbot && chatbot.classList.contains("chatbot-fullscreen")) {
                chatbot.classList.remove("chatbot-fullscreen");
                document.body.classList.remove("chatbot-fullscreen-open");
                const btn = document.getElementById("chatbot-fs-btn");
                setBtnState(btn, false);
            }
        }
    });

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", injectFullscreenBtn);
    } else {
        injectFullscreenBtn();
    }

    // 监听动态 DOM 变更并保持注入
    const obs = new MutationObserver(function() {
        injectFullscreenBtn();
    });
    obs.observe(document.documentElement, { childList: true, subtree: true });
})();
