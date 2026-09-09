(function () {
    const ENTER_FS_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>';
    const EXIT_FS_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14h6m0 0v6m0-6L3 21m17-7h-6m0 0v6m0-6l7 7M4 10h6m0 0V4m0 6L3 3m17 7h-6m0 0V4m0 6l7-7"/></svg>';

    function setBtnState(btn, isFull) {
        if (!btn) return;
        btn.innerHTML = isFull ? EXIT_FS_ICON : ENTER_FS_ICON;
        btn.title = isFull ? "退出网页全屏 (Esc)" : "网页全屏";
        btn.setAttribute("aria-label", btn.title);
    }

    function getFullscreenTarget() {
        const wrapper = document.getElementById("chat-interface-wrapper");
        if (wrapper) return wrapper;
        const chatbot = document.getElementById("main-chatbot");
        if (chatbot) {
            return chatbot.closest(".column") || chatbot;
        }
        return null;
    }

    function isFullscreenActive() {
        const target = getFullscreenTarget();
        return target ? (target.classList.contains("chat-wrapper-fullscreen") || target.classList.contains("chatbot-fullscreen")) : false;
    }

    function toggleFullscreen(forceState) {
        const target = getFullscreenTarget();
        if (!target) return;

        let isFull;
        if (typeof forceState === "boolean") {
            isFull = forceState;
            target.classList.toggle("chat-wrapper-fullscreen", isFull);
        } else {
            isFull = target.classList.toggle("chat-wrapper-fullscreen");
        }

        // 彻底锁定根容器滚动条，避免外层出现多余滚动条
        document.documentElement.classList.toggle("chatbot-fullscreen-open", isFull);
        document.body.classList.toggle("chatbot-fullscreen-open", isFull);

        const btn = document.getElementById("chatbot-fs-btn");
        setBtnState(btn, isFull);
    }

    function injectFullscreenBtn() {
        const chatbot = document.getElementById("main-chatbot");
        if (!chatbot) return false;

        let fsBtn = document.getElementById("chatbot-fs-btn");
        if (!fsBtn) {
            fsBtn = document.createElement("button");
            fsBtn.id = "chatbot-fs-btn";
            fsBtn.type = "button";
            fsBtn.className = "chatbot-fs-btn icon-button";
            const isFull = isFullscreenActive();
            setBtnState(fsBtn, isFull);

            fsBtn.onclick = function (e) {
                e.preventDefault();
                e.stopPropagation();
                toggleFullscreen();
            };
        }

        // 查找 Chatbot 右上角的按钮容器与清空按钮
        const allButtons = Array.from(chatbot.querySelectorAll("button:not(#chatbot-fs-btn)"));
        let targetBtn = allButtons.find(b => {
            const label = (b.getAttribute("aria-label") || b.title || "").toLowerCase();
            const cls = (b.className || "").toLowerCase();
            return label.includes("clear") || label.includes("trash") || label.includes("清空") || label.includes("delete") || (cls.includes("icon-button") && b.closest(".header"));
        });

        const header = chatbot.querySelector(".header") || (targetBtn ? targetBtn.parentElement : null);

        if (targetBtn && targetBtn.parentElement) {
            // 清空按钮存在时，精准插入到其左侧，并移除 fallback 类
            if (fsBtn.nextElementSibling !== targetBtn || fsBtn.parentElement !== targetBtn.parentElement) {
                fsBtn.classList.remove("chatbot-fs-btn-fallback");
                targetBtn.parentElement.insertBefore(fsBtn, targetBtn);
            }
            return true;
        }

        if (header) {
            // header 容器存在但尚未有清空按钮时，挂入 header 内
            if (fsBtn.parentElement !== header) {
                fsBtn.classList.remove("chatbot-fs-btn-fallback");
                header.appendChild(fsBtn);
            }
            return true;
        }

        // 兜底：若 header 容器尚未生成，暂挂在 chatbot 顶部，样式与位置与原生保持一致 (top: 6px, right: 6px)
        if (!fsBtn.isConnected || fsBtn.parentElement !== chatbot) {
            fsBtn.classList.add("chatbot-fs-btn-fallback");
            chatbot.style.position = "relative";
            chatbot.appendChild(fsBtn);
        }
        return true;
    }

    // -----------------------------------------------------------------
    // 智能流式输出吸底跟踪控制器 (Auto-Scroll Manager)
    // -----------------------------------------------------------------
    let autoScrollEnabled = true;
    let scrollContainer = null;
    let scrollBtn = null;

    function findScrollContainer() {
        const chatbot = document.getElementById("main-chatbot");
        if (!chatbot) return null;

        // 优先检查常见 Gradio 消息包装容器
        const selectors = [
            '[data-testid="chatbot-messages"]',
            '.bubble-wrap',
            '.wrapper',
            '.message-wrap',
            '.scroll-hide',
            'div:has(> [class*="message"])'
        ];

        for (const sel of selectors) {
            try {
                const el = chatbot.querySelector(sel);
                if (el && (el.scrollHeight > 0 || el.clientHeight > 0)) {
                    return el;
                }
            } catch (e) { }
        }

        // 递归查找首个 overflow-y 为 auto/scroll 的容器
        const allDivs = chatbot.querySelectorAll("div");
        for (const d of allDivs) {
            const style = window.getComputedStyle(d);
            if (style.overflowY === "auto" || style.overflowY === "scroll") {
                return d;
            }
        }

        return chatbot;
    }

    function createScrollToBottomBtn() {
        if (scrollBtn && scrollBtn.isConnected) return scrollBtn;
        const chatbot = document.getElementById("main-chatbot");
        if (!chatbot) return null;

        scrollBtn = document.createElement("button");
        scrollBtn.id = "chatbot-scroll-bottom-btn";
        scrollBtn.type = "button";
        scrollBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M19 12l-7 7-7-7"/></svg><span>最新</span>';
        scrollBtn.title = "滚动至最新回复";
        scrollBtn.style.cssText = `
            position: absolute;
            bottom: 16px;
            right: 20px;
            z-index: 50;
            display: none;
            align-items: center;
            gap: 4px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 600;
            color: #4338ca;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #c7d2fe;
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
            cursor: pointer;
            backdrop-filter: blur(6px);
            transition: all 0.2s ease;
        `;

        scrollBtn.onmouseenter = () => {
            scrollBtn.style.transform = "translateY(-2px)";
            scrollBtn.style.boxShadow = "0 6px 16px rgba(79, 70, 229, 0.25)";
        };
        scrollBtn.onmouseleave = () => {
            scrollBtn.style.transform = "translateY(0)";
            scrollBtn.style.boxShadow = "0 4px 12px rgba(79, 70, 229, 0.15)";
        };

        scrollBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            autoScrollEnabled = true;
            scrollToBottom(true);
            scrollBtn.style.display = "none";
        };

        chatbot.style.position = "relative";
        chatbot.appendChild(scrollBtn);
        return scrollBtn;
    }

    function scrollToBottom(force = false) {
        const container = scrollContainer || findScrollContainer();
        if (!container) return;

        if (force || autoScrollEnabled) {
            container.scrollTop = container.scrollHeight;
        }
    }

    function attachScrollListener() {
        const container = findScrollContainer();
        if (!container || container === scrollContainer) return;

        scrollContainer = container;
        // createScrollToBottomBtn();

        scrollContainer.addEventListener("scroll", function () {
            const threshold = 120; // 距离底部的容差像素
            const distanceToBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight;

            if (distanceToBottom > threshold) {
                // 用户主动往上翻看历史，暂停自动吸底，显示回到最新按钮
                autoScrollEnabled = false;
                if (scrollBtn) scrollBtn.style.display = "inline-flex";
            } else {
                // 用户回到接近底部，恢复自动吸底
                autoScrollEnabled = true;
                if (scrollBtn) scrollBtn.style.display = "none";
            }
        }, { passive: true });
    }

    // -----------------------------------------------------------------
    // 输入框操作图标定制：输入时回车键，运行时红色方块 (中断运行)
    // -----------------------------------------------------------------
    const ENTER_KEY_ICON = '<svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2.3" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 10 4 15 9 20"></polyline><path d="M20 4v7a4 4 0 0 1-4 4H4"></path></svg>';
    const STOP_SQUARE_ICON = '<svg viewBox="0 0 24 24" width="9" height="9" fill="#ef4444"><rect x="4" y="4" width="16" height="16" rx="3.5" ry="3.5"></rect></svg>';

    function customizeInputButtons() {
        // 1. 回车发送按钮图标定制
        const submitBtns = document.querySelectorAll('[data-testid="submit-button"], button.submit-button');
        submitBtns.forEach(btn => {
            if (btn.getAttribute("data-custom-icon") !== "enter") {
                btn.innerHTML = ENTER_KEY_ICON;
                btn.title = "发送需求 (Enter)";
                btn.setAttribute("aria-label", "发送");
                btn.setAttribute("data-custom-icon", "enter");
            }
        });

        // 2. 运行时红色方块中断按钮图标定制
        const stopBtns = document.querySelectorAll('[data-testid="stop-button"], button.stop-button');
        stopBtns.forEach(btn => {
            if (btn.getAttribute("data-custom-icon") !== "stop") {
                btn.innerHTML = STOP_SQUARE_ICON;
                btn.title = "中断 Agent 运行 (Stop)";
                btn.setAttribute("aria-label", "中断运行");
                btn.setAttribute("data-custom-icon", "stop");
            }
        });
    }

    // -----------------------------------------------------------------
    // 历史会话列表交互分发 (Session Select & Delete Handlers)
    // -----------------------------------------------------------------
    window.handleSessionClick = function (e, sid) {
        if (!sid) return;
        if (e && e.target && e.target.closest(".session-del-btn")) {
            return;
        }
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }

        window.__target_select_session_id = sid;
        const triggerBtn = document.getElementById("target-select-btn") || document.querySelector("#target-select-btn button");
        if (triggerBtn) {
            triggerBtn.click();
        }
    };

    window.handleSessionDelete = function (e, sid) {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        if (!sid) return;

        window.__target_delete_session_id = sid;
        const triggerBtn = document.getElementById("target-delete-btn") || document.querySelector("#target-delete-btn button");
        if (triggerBtn) {
            triggerBtn.click();
        }
    };

    // 监听 ESC 键退出全屏
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
            if (isFullscreenActive()) {
                toggleFullscreen(false);
            }
        }
    });

    function init() {
        injectFullscreenBtn();
        attachScrollListener();
        customizeInputButtons();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    // 监听动态 DOM 变更：保持按钮注入、图标更新，并进行流式吸底
    let scrollThrottleTimer = null;
    const obs = new MutationObserver(function () {
        injectFullscreenBtn();
        attachScrollListener();
        customizeInputButtons();

        // 当内容变动（Agent 流式生成）时，触发吸底
        if (autoScrollEnabled) {
            if (!scrollThrottleTimer) {
                scrollThrottleTimer = requestAnimationFrame(() => {
                    scrollToBottom(false);
                    scrollThrottleTimer = null;
                });
            }
        }
    });

    obs.observe(document.documentElement, {
        childList: true,
        subtree: true,
        characterData: true
    });
})();
