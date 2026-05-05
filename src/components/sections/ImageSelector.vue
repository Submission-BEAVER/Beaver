<script setup>
import { ref, onMounted } from 'vue';

const visualizations = [
  {
    label: "Financial QA",
    thumbnail: "./images/financial-qa.png",
    preview: "./images/financial-qa.png"
  },
  {
    label: "GovReport",
    thumbnail: "./images/govreport.png",
    preview: "./images/govreport.png"
  },
  {
    label: "GSM100",
    thumbnail: "./images/gsm100.png",
    preview: "./images/gsm100.png"
  },
  {
    label: "CodeU",
    thumbnail: "./images/codeu.png",
    preview: "./images/codeu.png"
  }
];

const outputImagesPaths = [];

let outputImagePath = ref("");
let indexSelected = ref(0);
let isLoading = ref(true);

const preloadImageSelector = () => {
  const promises = [];
      
  for (let i = 0; i < visualizations.length; i++) {
    const outputImg = new Image();
    outputImagesPaths[i] = outputImg;
    const outputPromise = new Promise((resolve) => {
        outputImg.onload = resolve;
    });
    outputImg.src = visualizations[i].preview;
    promises.push(outputPromise);
  }

  return Promise.all(promises).then(() => {
    isLoading.value = false;
  });
}

onMounted(() => {
    preloadImageSelector();
    handleChange(0);
});

const handleChange = (value) => {
  indexSelected.value = value;
  outputImagePath.value = outputImagesPaths[value].src;
};

</script>

<template>
  <section class="section-shell selector-shell">
    <h1 class="section-title">Qualitative Visualization</h1>

    <div class="selector-layout">
      <div class="thumb-row">
        <button
          class="thumb-card"
          v-for="(visualization, index) in visualizations"
          :key="visualization.label"
          type="button"
          @click="handleChange(index)"
          :class="{ 'selected-card': indexSelected === index }"
        >
          <el-image
            class="image"
            :src="visualization.thumbnail"
            fit="contain"
          />
          <span class="thumb-label">{{ visualization.label }}</span>
        </button>
      </div>

      <el-skeleton
        class="preview-skeleton-container"
        :loading="isLoading"
        animated
        :throttle="1000"
      >
        <template #template>
          <el-skeleton-item variant="image" class="preview-skeleton" />
        </template>
        <template #default>
          <div class="preview-frame">
            <img :src="outputImagePath" class="output-image" />
          </div>
        </template>
      </el-skeleton>
    </div>
  </section>
</template>

<style scoped>

.selector-shell {
  max-width: 1120px;
  padding-bottom: 44px;
}

.selector-layout {
  margin-top: 8px;
}

.thumb-row {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 0 auto 18px;
}

.thumb-card {
  align-items: center;
  background: #ffffff;
  border: 1px solid #dce6ef;
  border-radius: 8px;
  color: #2f3d4c;
  cursor: pointer;
  display: grid;
  gap: 10px;
  grid-template-columns: 54px minmax(0, 1fr);
  min-height: 76px;
  padding: 10px;
  text-align: left;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.thumb-card:hover {
  border-color: #85aebe;
  box-shadow: 0 10px 24px rgba(15, 35, 55, 0.09);
  transform: translateY(-1px);
}

.selected-card {
  border-color: #0f5f78;
  box-shadow: 0 12px 26px rgba(15, 95, 120, 0.15);
}

.image {
  aspect-ratio: 1;
  border: 1px solid #e5edf4;
  border-radius: 6px;
  overflow: hidden;
  width: 54px;
}

.thumb-label {
  font-size: 15px;
  font-weight: 800;
  line-height: 1.2;
}

.preview-skeleton-container {
  width: 100%;
}

.preview-skeleton {
  aspect-ratio: 4 / 3;
  max-height: 90vh;
  width: 100%;
}

.preview-frame {
  background: #f8fafc;
  border: 1px solid #dce6ef;
  border-radius: 8px;
  box-shadow: 0 14px 34px rgba(15, 35, 55, 0.10);
  padding: 14px;
}

.output-image {
  background: #ffffff;
  border-radius: 6px;
  display: block;
  height: auto;
  max-height: 1680px;
  margin: 0 auto;
  object-fit: contain;
  width: 100%;
}

@media (max-width: 900px) {
  .thumb-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .thumb-row {
    grid-template-columns: 1fr;
  }

  .preview-frame {
    padding: 8px;
  }
}

</style>
