import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { SnackbarComponent } from '../snackbar/snackbar.component';
import { MatSnackBar } from '@angular/material/snack-bar';
import { stringify } from 'querystring';

@Component({
  selector: 'app-community',
  templateUrl: './community.component.html',
  styleUrls: ['../getting-started/getting-started.component.css']
})
export class CommunityComponent implements OnInit {

  constructor(private route: ActivatedRoute, private _snackBar: MatSnackBar) { }

  ngOnInit() {
    this.route.queryParams.subscribe(params => {
      this.showMessage(params.slackEmailResponse);
    });
  }

  showMessage(code) {
    var message:string;
    if(code=="250"){
      message = "Successfully sent out the slack channel form";
    }
    else{
      message = "code: " + code +  ", cannot sent out the slack channel form";
    }

    this._snackBar.openFromComponent(SnackbarComponent, {
      duration: 5000,
      data: message
    });
  }
}
