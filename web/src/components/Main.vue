<template>
  <div class="container">
    <div class="text-container">
      <el-input
        v-model="textarea1"
        type="textarea"
        rows="35"
        placeholder="Please input"
      />
      <div class="input-tool">
        <el-button type="primary" :disabled="textarea1.length == 0" @click="onReset()">Reset</el-button>
      </div>
    </div>

    <div class="summary-tool">
      <el-text type="primary" tag="b">Summary percent</el-text>
      <el-slider v-model="keep"></el-slider>
      <el-button type="primary" round :disabled="textarea1.length == 0" @click="onSendRequest()">Summarize</el-button>
    </div>

    <div class="text-container">
      <el-input
        v-model="textarea2"
        type="textarea"
        rows="35"
        placeholder="Summarized text"
        disabled
      />
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import { summarize } from '../services/summarize'
const textarea1 = ref('')
const textarea2 = ref('')
const keep = ref(30)

const onReset = () => {
  textarea1.value = ""
  textarea2.value = ""
}

const onSendRequest = async () => {
    console.log(textarea1.value)
    const res = await summarize(textarea1.value, keep.value/100)
    console.log(res)
    // textarea2.value = res.data.summary
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
