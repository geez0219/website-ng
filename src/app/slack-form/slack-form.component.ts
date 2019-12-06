import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-slack-form',
  templateUrl: './slack-form.component.html',
  styleUrls: ['./slack-form.component.css']
})
export class SlackFormComponent implements OnInit {

  testvalue: any;
  hasError:boolean;
  disabled:boolean = true;
  otherPurposeValue:string = "";
  constructor() { }

  ngOnInit() {
  }

  onClick(content){
    this.testvalue=content;
  }

  log(content){
    console.log(content);
  }
  validateAndSubmit(formControlList, form){
    console.log(formControlList);

    for (var i=0;i<formControlList.length; i++){
      if(!formControlList[i].valid){
        this.hasError = true;
        return ;
      }
    }

    form.submit();
  }

  toggle(){
    this.disabled = !this.disabled;
  }

  clickRadio(x:boolean){
    if(!x){
      this.otherPurposeValue = "";
    }
    this.disabled = !x;
  }
}
