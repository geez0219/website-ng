import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-slack-form',
  templateUrl: './slack-form.component.html',
  styleUrls: ['./slack-form.component.css']
})
export class SlackFormComponent implements OnInit {
  hasError:boolean;
  disabledPurpose:boolean = true;
  disabledExp:boolean = true;
  otherPurposeValue:string = "";
  otherExpValue:string = "";
  constructor() { }

  ngOnInit() {
  }

  log(content){
    console.log(content);
  }

  validateAndSubmit(boolList, form){
  // boolList need to be all true to passs the validation
    for (var i=0;i<boolList.length; i++){
      if(!boolList[i]){
        this.hasError = true;
        return ;
      }
    }
    form.submit();
  }

  clickRadioPurpose(x:boolean){
    if(!x){
      this.otherPurposeValue = "";
    }
    this.disabledPurpose = !x;
  }

  clickRadioExp(x:boolean){
    if(!x){
      this.otherExpValue = "";
    }
    this.disabledExp = !x;
  }

}
