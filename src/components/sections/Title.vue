<script lang="ts" setup>

import { ElIcon } from 'element-plus'
import { Files, MagicStick } from '@element-plus/icons-vue'

// logo地址，没有则置为""即可
const logo = './images/beaver1.jpeg'

// 标题
const title = 'BEAVER: A Training-Free Hierarchical Prompt Compression Method via Structure-Aware Page Selection'

// 强调内容
const emphases = [
  {
    label: "Structure-aware",
    text: "Hierarchical page selection preserves discourse integrity."
  },
  {
    label: "26.9x faster",
    text: "Compresses one million tokens in 10.4 seconds."
  },
  {
    label: "Training-free SOTA",
    text: "Strong results on RULER, L-Eval, LongBench, and ZeroSCROLLS."
  },
  {
    label: "Plug-and-play",
    text: "Works across model sizes without parameter updates."
  }
]

// 提供引导资料链接
const buttons = [
  {
    disabled: false,
    name: "Code",
    link: "https://github.com/Submission-BEAVER/Beaver",
    component: Files,
  },
  {
    disabled: true,
    name: "Demo",
    link: "#",
    component: MagicStick,
  },
]

const linkedButtons = buttons.filter((button) => !button.disabled && button.link !== '#')
const disabledButtons = buttons.filter((button) => button.disabled || button.link === '#')

</script>

<template>
  <section class="hero-section">
    <div class="notice">
      Project code is available on GitHub.
    </div>

    <div class="hero-inner">
      <div class="hero-logo-wrap" v-if="logo">
        <el-image :src="logo" class="logo" fit="cover" />
      </div>

      <p class="venue">NeurIPS 2026 Project Page</p>
      <h1 class="paper-title">{{ title }}</h1>

      <div class="emphasis-grid">
        <div class="emphasis-card" v-for="emphasis in emphases" :key="emphasis.label">
          <span class="emphasis-label">{{ emphasis.label }}</span>
          <span class="emphasis-text">{{ emphasis.text }}</span>
        </div>
      </div>

      <div class="guidance-row">
        <a
          v-for="button in linkedButtons"
          :key="button.name"
          :href="button.link"
          target="_blank"
          rel="noreferrer noopener"
        >
          <el-button class="guidance-button" size="large">
            <el-icon :size="17">
              <component :is="button.component" />
            </el-icon>
            <span class="btn-text">{{ button.name }}</span>
          </el-button>
        </a>
        <el-button
          class="guidance-button"
          size="large"
          :disabled="button.disabled"
          v-for="button in disabledButtons"
          :key="button.name"
        >
          <el-icon :size="17">
            <component :is="button.component" />
          </el-icon>
          <span class="btn-text">{{ button.name }}</span>
        </el-button>
      </div>
    </div>
  </section>
</template>

<style scoped>

.hero-section {
  border-bottom: 1px solid #e7edf3;
  background:
    linear-gradient(180deg, #f8fbfd 0%, #ffffff 78%),
    radial-gradient(circle at 20% 10%, rgba(54, 116, 181, 0.08), transparent 36%);
}

.notice {
  background: #102235;
  color: #f6fbff;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.4;
  padding: 10px 24px;
  text-align: center;
}

.hero-inner {
  max-width: 1180px;
  margin: 0 auto;
  padding: 40px 24px 36px;
  text-align: center;
}

.hero-logo-wrap {
  display: flex;
  justify-content: center;
  margin-bottom: 18px;
}

.logo {
  width: 112px;
  height: 112px;
  border: 1px solid #d8e2eb;
  border-radius: 50%;
  box-shadow: 0 12px 32px rgba(25, 48, 71, 0.14);
}

.venue {
  color: #44718a;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0;
  margin: 0 0 12px;
  text-transform: uppercase;
}

.paper-title {
  color: #111827;
  font-family: "MyFont", Georgia, serif;
  font-size: clamp(32px, 4vw, 54px);
  font-weight: 800;
  letter-spacing: 0;
  line-height: 1.08;
  margin: 0 auto;
  max-width: 1050px;
  text-align: center;
}

.emphasis-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 28px auto 24px;
  max-width: 1080px;
}

.emphasis-card {
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid #dce6ef;
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(20, 43, 68, 0.07);
  padding: 16px 14px;
  text-align: left;
}

.emphasis-label {
  color: #0f5f78;
  display: block;
  font-size: 15px;
  font-weight: 800;
  margin-bottom: 6px;
}

.emphasis-text {
  color: #44515f;
  display: block;
  font-size: 14px;
  line-height: 1.45;
}

.guidance-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
}

.guidance-button {
  border-radius: 999px;
  border-color: #27394a;
  color: #172535;
  font-weight: 700;
  min-width: 112px;
}

.guidance-button:hover {
  background: #14283a;
  border-color: #14283a;
  color: #ffffff;
}

.btn-text {
  font-size: 15px;
}

/* 姓名和地址按钮 */
a:-webkit-any-link {
  text-decoration: none;
}

/* 取消鼠标焦点悬浮在链接上的颜色装饰 */
a:hover {
  color: inherit;
  border-bottom: none;
}

/* 链接装饰，取消下划线和链接颜色 */
a {
	text-decoration: None;
	color: inherit;
}

@media (max-width: 900px) {
  .hero-inner {
    padding: 32px 18px 30px;
  }

  .emphasis-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .notice {
    font-size: 13px;
    padding: 9px 14px;
  }

  .logo {
    height: 92px;
    width: 92px;
  }

  .paper-title {
    font-size: 29px;
  }

  .emphasis-grid {
    grid-template-columns: 1fr;
    margin-top: 22px;
  }
}

</style>
