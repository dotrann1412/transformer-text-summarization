<template>
  <div class="container">
    <div class="text-container">
      <el-input
        v-model="textarea1"
        type="textarea"
        rows="35"
        placeholder="Input text or upload file"
        :disabled="isSummarizing"
      />
      <div class="input-tool">
        <el-upload 
          :action="upload_api" 
          :show-file-list="false" 
          method="post"
          name="filename"
          :on-success="onUploadSuccess"
          :disabled="isSummarizing">
          <el-button type="primary">Upload</el-button>
        </el-upload>
        <el-button
          type="primary" 
          :disabled="textarea1.length == 0 || isSummarizing" 
          @click="onReset()" 
          style="margin-left: 10px;">
          Reset
        </el-button>
      </div>
    </div>

    <div class="summary-tool">
      <el-text type="primary" tag="b" truncated>Summary percent</el-text>
      <el-slider v-model="keep"></el-slider>
      <el-button 
        type="primary" 
        round 
        :disabled="textarea1.length == 0" 
        @click="onSendRequest()"
        :loading="isSummarizing"
        style="width: 8vw;">
          Summarize
      </el-button>
    </div>

    <div class="text-container">
      <el-input
        v-model="textarea2"
        type="textarea"
        rows="35"
        placeholder="Summarized text"
        readonly
      />
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import type { UploadProps, UploadUserFile } from 'element-plus'
import { summarize } from '../services/summarize'
import { ocrSpace } from 'ocr-space-api-wrapper'

const textarea1 = ref('')
const textarea2 = ref('')
const keep = ref(30)

// reset feature
const onReset = () => {
  textarea1.value = ""
  textarea2.value = ""
}

// upload feature
const upload_api = ref(import.meta.env.VITE_URL + '/cores/upload/')
const onUploadSuccess: UploadProps['onSuccess'] = (response) => {
  textarea1.value = response['ocr']
}

// summarize feature
const isSummarizing = ref(false)
const onSendRequest = async () => {
  textarea2.value = ""
  isSummarizing.value = true
  const res = await summarize(textarea1.value, keep.value/100)
  if(res.data.summary != undefined)
    textarea2.value = res.data.summary
  else
    textarea2.value = 'An error occured! Please try again.'
  isSummarizing.value = false
}

</script>

<style scoped lang="scss">
.container {
  margin: auto;
  position: relative;
  display: flex;
  flex-direction: row;
  align-items: center;
  .text-container {
    padding: 30px;
    width: 45vw;
  }
  .input-tool {
    margin-top: 1vh;
    display: flex;
    flex-direction: row;
  }
}
</style>

<style lang="scss">
// disable resize in textarea
.text-container .ep-textarea .ep-textarea__inner {
    resize: none
}
</style>